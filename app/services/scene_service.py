import asyncio
import time as pytime
from collections import deque
from typing import List, Dict, Any, Optional
from app.services.base import BaseService
from app.services.detect_service import DetectService
from app.services.caption_service import CaptionService
from app.services.florence_service import FlorenceService
from app.services.narration_service import NarrationService
from app.schemas.scene import SceneAnalysisResult, HazardItem
from app.schemas.detect import DetectionItem
from app.schemas.perception import PerceptionGraph, GroundedObject
from app.utils.tracker import TemporalConsistencyTracker
from app.core.config import settings
import logging

logger = logging.getLogger("accessvision")

class SceneService(BaseService):
    """Orchestrates overall scene understanding, combining object detection,
    captioning, hazard analysis, and accessibility narration. Maintains a rolling memory
    of recent scene states and uses a temporal consistency tracker.
    """

    # Session state cache: session_id -> dict containing session data (Phase 3)
    _sessions = {}

    @classmethod
    def _get_session_state(cls, session_id: str) -> dict:
        current_time = pytime.time()
        
        # Clean up expired sessions (older than 10 minutes)
        expired_sessions = [
            sid for sid, state in cls._sessions.items()
            if current_time - state.get("last_activity", 0) > 600
        ]
        for sid in expired_sessions:
            cls._sessions.pop(sid, None)
            
        if session_id not in cls._sessions:
            cls._sessions[session_id] = {
                "tracker": TemporalConsistencyTracker(max_inactive_frames=3, iou_threshold=0.3),
                "memory": deque(maxlen=10),
                "cached_image_gray": None,
                "cached_detections": None,
                "cached_perception_graph": None,
                "cached_caption": None,
                "cached_narration": None,
                "cached_hazards": None,
                "cached_time": 0.0,
                "last_activity": current_time
            }
        else:
            cls._sessions[session_id]["last_activity"] = current_time
            
        return cls._sessions[session_id]

    def __init__(self):
        from app.services.crop_service import CropService
        from app.services.color_service import ColorService
        self.detect_service = DetectService()
        self.caption_service = CaptionService()
        self.florence_service = FlorenceService()
        self.narration_service = NarrationService()
        self.crop_service = CropService()
        self.color_service = ColorService()

    def _assess_hazards(self, detections: List[DetectionItem]) -> List[HazardItem]:
        """Determines if any of the detected objects pose a hazard based on type and proximity (bounding box height)."""
        hazards = []
        
        # Hazard classes lists
        burn_hazards = {"oven", "microwave", "toaster", "stove", "hot dog", "cup"}
        tripping_hazards = {"chair", "backpack", "suitcase", "dog", "cat", "shoes", "vase", "skateboard"}
        collision_hazards = {"bed", "dining table", "tv", "refrigerator", "bench", "couch"}
        sharp_hazards = {"scissors", "knife", "fork"}

        for d in detections:
            label = d.label.lower()
            xmin, ymin, xmax, ymax = d.box
            
            # Bounding box coordinates:
            # We check if coordinate ymax is in the bottom region of the image
            is_normalized = all(0.0 <= c <= 1.0 for c in d.box)
            proximity_threshold = 0.65 if is_normalized else 640 * 0.65
            
            is_near = ymax > proximity_threshold
            
            hazard_type = None
            severity = "low"
            description = ""

            if label in sharp_hazards:
                hazard_type = "sharp_object_risk"
                severity = "high" if is_near else "medium"
                description = f"A sharp {label} is detected in your field of view."
            elif label in burn_hazards:
                hazard_type = "burn_risk"
                severity = "high" if is_near else "medium"
                description = f"A potential hot {label} is detected nearby."
            elif label in tripping_hazards:
                hazard_type = "tripping_risk"
                severity = "high" if is_near else "medium"
                description = f"A {label} is on the floor, posing a tripping hazard."
            elif label in collision_hazards and is_near:
                hazard_type = "collision_risk"
                severity = "medium"
                description = f"A large {label} is directly in front of you."

            if hazard_type:
                hazards.append(HazardItem(
                    label=d.label,
                    box=d.box,
                    type=hazard_type,
                    severity=severity,
                    description=description
                ))
                
        return hazards

    async def analyze_scene(
        self, 
        image_bytes: bytes, 
        detections: Optional[List[DetectionItem]] = None, 
        is_mirrored: bool = False,
        mode: str = "fast",
        session_id: Optional[str] = None
    ) -> SceneAnalysisResult:
        """Runs multi-speed scene understanding based on requested speed mode and temporal cache status."""
        logger.info(f"Starting scene analysis in mode: {mode} (session: {session_id})")
        from app.core.telemetry import trace_stage, get_current_telemetry
        import numpy as np
        
        telemetry = get_current_telemetry()
        
        sid = session_id or "global_default"
        state = self._get_session_state(sid)
        
        caption = ""
        florence_caption = ""
        blip_caption = ""
        
        # Track latencies
        yolo_latency = 0.0
        blip_latency = 0.0
        florence_latency = 0.0
        
        # Load image for service usage
        from app.utils.image import load_image_from_bytes
        pil_image = load_image_from_bytes(image_bytes)
        w, h = pil_image.size
        
        # Grayscale downscaled array for visual similarity check (Phase 3)
        curr_gray = np.array(pil_image.convert("L").resize((32, 32)), dtype=np.float32)
        
        is_static = False
        is_similar = False
        florence_timeout_occurred = False
        
        try:
            with self.measure_latency() as metrics:
                # 1. Run YOLO (Fast Loop Core)
                if detections is None:
                    t_start = pytime.perf_counter()
                    detect_res = await self.detect_service.detect_objects(pil_image)
                    detections = detect_res.detections if detect_res.success else []
                    yolo_latency = (pytime.perf_counter() - t_start) * 1000
                else:
                    logger.info("[CACHE] Reused detections in scene analysis")

                # 2. Check scene similarity with cached frame (Phase 3)
                current_time = asyncio.get_event_loop().time()
                cached_gray = state["cached_image_gray"]
                cached_time = state["cached_time"]
                cache_age = current_time - cached_time if cached_time > 0 else float('inf')
                
                if cached_gray is not None and cache_age < 5.0:
                    diff = np.mean(np.abs(curr_gray - cached_gray))
                    if diff < settings.SCENE_STATIC_THRESHOLD:
                        is_static = True
                        logger.info(f"[CACHE] Scene similarity diff: {diff:.2f} < STATIC threshold {settings.SCENE_STATIC_THRESHOLD}. Strictly static scene.")
                    elif diff < settings.SCENE_SIMILARITY_THRESHOLD:
                        cached_dets = state["cached_detections"]
                        if cached_dets is not None:
                            is_similar = self._check_detection_stability(detections, cached_dets, settings.SCENE_IOU_THRESHOLD)
                            logger.info(f"[CACHE] Scene similarity diff: {diff:.2f} < SIMILARITY threshold {settings.SCENE_SIMILARITY_THRESHOLD}. Detections stable: {is_similar}.")
                
                # 3. Resolve caption based on mode and temporal cache (Phase 3 & 5)
                if mode != "fast" and not (is_static or is_similar):
                    run_blip = settings.MIGRATION_STAGE < 3
                    run_florence = True
                    
                    tasks = []
                    if run_blip:
                        tasks.append(self.caption_service.generate_caption(image_bytes))
                    if run_florence:
                        tasks.append(self.florence_service.get_detailed_caption(pil_image))
                        
                    try:
                        results = await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=settings.FLORENCE_TIMEOUT
                        )
                        
                        idx = 0
                        if run_blip:
                            blip_res = results[idx]
                            if not isinstance(blip_res, Exception) and blip_res.success:
                                blip_caption = blip_res.caption
                                blip_latency = blip_res.metrics.get("inference_ms", 0.0)
                            idx += 1
                        if run_florence:
                            florence_res = results[idx]
                            if not isinstance(florence_res, Exception):
                                florence_caption = florence_res
                                if telemetry and "FLORENCE" in telemetry.timings:
                                    florence_latency = telemetry.timings["FLORENCE"]
                            idx += 1
                            
                        # Side-by-Side comparison logging
                        logger.info(
                            f"[MIGRATION EVAL] BLIP Caption: '{blip_caption}' ({blip_latency:.1f}ms) "
                            f"| Florence-2 Caption: '{florence_caption}'"
                        )
                        
                        # Choose caption for narration
                        if settings.MIGRATION_STAGE == 1:
                            caption = blip_caption
                        else:
                            caption = florence_caption if florence_caption else blip_caption
                            
                    except asyncio.TimeoutError:
                        logger.warning("[FALLBACK] Florence/BLIP captioning timed out. Falling back to fast narration.")
                        florence_timeout_occurred = True
                        caption = ""
                elif is_static or is_similar:
                    logger.info("[CACHE] Reused perception graph")
                    caption = state["cached_caption"]
                else:
                    logger.info("[SCHEDULER] Fast Loop: Skipping captioning models.")
                    caption = ""
 
                # 4. Slow Loop deep grounding: get Florence-2 objects if mode is slow and no cache hit
                florence_objects = {"bboxes": [], "labels": []}
                if mode == "slow" and not (is_static or is_similar) and not florence_timeout_occurred:
                    logger.info("[SCHEDULER] Slow Loop: Running Florence-2 <OD> task")
                    try:
                        florence_objects = await asyncio.wait_for(
                            self.florence_service.get_objects(pil_image),
                            timeout=settings.FLORENCE_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[FALLBACK] Florence <OD> task timed out. Switched to fast mode.")
                        florence_timeout_occurred = True

                # 5. Handle cache hits and timeout fallbacks (Phase 3 & 5)
                if is_static or is_similar:
                    logger.info("[CACHE] Reused Florence caption and narration")
                    perception_graph = state["cached_perception_graph"]
                    narration = state["cached_narration"]
                    hazards = state["cached_hazards"]
                elif florence_timeout_occurred:
                    logger.warning("[FALLBACK] Switched to fast grounded narration after timeout")
                    hazards = self._assess_hazards(detections)
                    narration = self.narration_service.generate_fallback_narration(
                        detections=detections,
                        hazards=hazards,
                        pil_image=pil_image
                    )
                    
                    # Build fast YOLO-only perception graph
                    current_colors = []
                    for d in detections:
                        with trace_stage("COLOR_CROP"):
                            cropped_pil = self.crop_service.crop_object(pil_image, d.box)
                        try:
                            color_res = self.color_service.analyze_color(cropped_pil, d.confidence, fast_mode=True)
                            current_colors.append(color_res["color_name"])
                        except Exception:
                            current_colors.append("unknown")
                        finally:
                            cropped_pil.close()
                            
                    current_time = asyncio.get_event_loop().time()
                    tracked_results = state["tracker"].update([
                        {"label": d.label, "box": d.box, "confidence": d.confidence, "source": "YOLO"}
                        for d in detections
                    ], current_colors, current_time)
                    
                    perception_objects = []
                    for track in tracked_results:
                        xmin, ymin, xmax, ymax = track["box"]
                        center_x = (xmin + xmax) / 2.0 / w
                        ymax_norm = ymax / h
                        area = ((xmax - xmin) * (ymax - ymin)) / (w * h)
                        
                        x_pos = "left" if center_x < 0.33 else ("right" if center_x > 0.66 else "center")
                        
                        if area >= 0.20:
                            depth = "foreground"
                            size_desc = "large"
                        elif ymax_norm > 0.65:
                            depth = "foreground"
                            size_desc = "medium"
                        elif area < 0.04:
                            depth = "background"
                            size_desc = "small"
                        else:
                            depth = "nearby"
                            size_desc = "medium"
                            
                        if x_pos == "center":
                            spatial_phrase = f"in the center {depth}" if depth != "nearby" else "nearby in the center"
                        else:
                            spatial_phrase = f"in the {depth} to your {x_pos}" if depth != "nearby" else f"to your {x_pos}"
                            
                        perception_objects.append(GroundedObject(
                            class_name=track["label"],
                            confidence=track["confidence"],
                            color=track["color"],
                            position=spatial_phrase,
                            size=size_desc,
                            grounding_source="YOLO",
                            narration_confidence="MEDIUM"
                        ))
                    perception_graph = PerceptionGraph(objects=perception_objects)
                else:
                    # Normal pipeline run (cache miss, no timeout)
                    # Construct Grounded Objects list (Fusion of YOLO and Florence-2)
                    raw_objects = []
                    
                    # Parse YOLO objects
                    for d in detections:
                        raw_objects.append({
                            "label": d.label,
                            "box": d.box,
                            "confidence": d.confidence,
                            "source": "YOLO"
                        })
                        
                    # Parse Florence-2 OD objects in slow mode
                    if mode == "slow":
                        f_bboxes = florence_objects.get("bboxes", [])
                        f_labels = florence_objects.get("labels", [])
                        for box, label in zip(f_bboxes, f_labels):
                            raw_objects.append({
                                "label": label,
                                "box": box,
                                "confidence": 0.65,
                                "source": "Florence"
                            })
                            
                    # Fuse objects (deduplicate overlapping boxes of the same class)
                    fused_objects = self._fuse_detections(raw_objects)
                    
                    # Fetch colors for current detections
                    current_colors = []
                    for obj in fused_objects:
                        with trace_stage("COLOR_CROP"):
                            cropped_pil = self.crop_service.crop_object(pil_image, obj["box"])
                        try:
                            color_res = self.color_service.analyze_color(cropped_pil, obj["confidence"], fast_mode=(mode == "fast"))
                            current_colors.append(color_res["color_name"])
                        except Exception:
                            current_colors.append("unknown")
                        finally:
                            cropped_pil.close()
                            
                    # Temporal Consistency Tracker
                    current_time = asyncio.get_event_loop().time()
                    tracked_results = state["tracker"].update(fused_objects, current_colors, current_time)
                    logger.info(f"[SCENE] Temporal consistency tracker active: {len(tracked_results)} stable objects.")
                    
                    # Build Perception Graph
                    perception_objects = []
                    for track in tracked_results:
                        xmin, ymin, xmax, ymax = track["box"]
                        center_x = (xmin + xmax) / 2.0 / w
                        ymax_norm = ymax / h
                        area = ((xmax - xmin) * (ymax - ymin)) / (w * h)
                        
                        x_pos = "left" if center_x < 0.33 else ("right" if center_x > 0.66 else "center")
                        
                        if area >= 0.20:
                            depth = "foreground"
                            size_desc = "large"
                        elif ymax_norm > 0.65:
                            depth = "foreground"
                            size_desc = "medium"
                        elif area < 0.04:
                            depth = "background"
                            size_desc = "small"
                        else:
                            depth = "nearby"
                            size_desc = "medium"
                            
                        if x_pos == "center":
                            spatial_phrase = f"in the center {depth}" if depth != "nearby" else "nearby in the center"
                        else:
                            spatial_phrase = f"in the {depth} to your {x_pos}" if depth != "nearby" else f"to your {x_pos}"
                            
                        # Grounding source
                        source = "YOLO"
                        for fo in fused_objects:
                            if fo["label"] == track["label"] and self._calculate_iou(fo["box"], track["box"]) > 0.5:
                                source = fo["source"]
                                break
                                
                        # Narration confidence tier
                        if track["confidence"] >= 0.75:
                            tier = "HIGH"
                        elif track["confidence"] >= 0.50:
                            tier = "MEDIUM"
                        else:
                            tier = "LOW"
                            
                        perception_objects.append(GroundedObject(
                            class_name=track["label"],
                            confidence=track["confidence"],
                            color=track["color"],
                            position=spatial_phrase,
                            size=size_desc,
                            grounding_source=source,
                            narration_confidence=tier
                        ))
                        
                    perception_graph = PerceptionGraph(objects=perception_objects)
                    logger.info(f"[SCENE] Perception graph generated with {len(perception_objects)} nodes.")
 
                    # Assess hazards
                    hazards = self._assess_hazards(detections)
 
                    # Generate narration using perception graph
                    narration = self.narration_service.generate_narration(
                        caption=caption,
                        detections=detections,
                        hazards=hazards,
                        pil_image=pil_image,
                        recent_memory=list(state["memory"]),
                        perception_graph=perception_graph
                    )
                    
                    # Update cache variables since it was a successful, fresh run
                    state["cached_image_gray"] = curr_gray
                    state["cached_detections"] = detections
                    state["cached_perception_graph"] = perception_graph
                    state["cached_caption"] = caption
                    state["cached_narration"] = narration
                    state["cached_hazards"] = hazards
                    state["cached_time"] = current_time
 
        finally:
            pil_image.close()
 
        result = SceneAnalysisResult(
            success=True,
            caption=caption,
            objects=detections,
            hazards=hazards,
            narration=narration,
            perception_graph=perception_graph,
            metrics={"inference_ms": metrics["latency_ms"]}
        )
 
        # Store in rolling scene memory
        state["memory"].append({
            "timestamp": asyncio.get_event_loop().time(),
            "caption": caption,
            "detections": [d.model_dump() for d in detections],
            "hazards": [h.model_dump() for h in hazards],
            "narration": narration
        })

        return result

    def _check_detection_stability(self, curr_dets: List[DetectionItem], cached_dets: List[DetectionItem], iou_thresh: float) -> bool:
        """Determines if the detected objects remain stable frame-to-frame (Phase 3)."""
        if len(curr_dets) != len(cached_dets):
            return False
        
        matched = 0
        for cd in curr_dets:
            for prd in cached_dets:
                if cd.label.lower() == prd.label.lower():
                    iou = self._calculate_iou(cd.box, prd.box)
                    if iou >= iou_thresh:
                        matched += 1
                        break
        return matched == len(curr_dets)

    def _fuse_detections(self, raw_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combines overlapping boxes of the same class from different models."""
        sorted_objs = sorted(raw_objects, key=lambda x: x["confidence"], reverse=True)
        fused = []
        
        for obj in sorted_objs:
            duplicate = False
            for f in fused:
                if f["label"].lower() == obj["label"].lower():
                    iou = self._calculate_iou(f["box"], obj["box"])
                    if iou > 0.5:
                        duplicate = True
                        if obj["source"] != f["source"]:
                            f["source"] = "YOLO+Florence"
                        break
            if not duplicate:
                fused.append(obj)
        return fused

    def _calculate_iou(self, b1: List[float], b2: List[float]) -> float:
        """Calculates Intersection over Union of two bounding boxes."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        inter = w * h
        
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        
        return inter / union if union > 0 else 0.0

    @classmethod
    def get_recent_memory(cls, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns the rolling memory queue contents for the given session."""
        sid = session_id or "global_default"
        state = cls._get_session_state(sid)
        return list(state["memory"])
