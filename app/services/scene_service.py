import asyncio
import time as pytime
import json
from collections import deque
from typing import List, Dict, Any, Optional, AsyncGenerator
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
        from app.services.fast_narration_service import FastNarrationService
        self.detect_service = DetectService()
        self.caption_service = CaptionService()
        self.florence_service = FlorenceService()
        self.narration_service = NarrationService()
        self.crop_service = CropService()
        self.color_service = ColorService()
        self.fast_narration_service = FastNarrationService()

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
        from app.services.enrichment_cache import enrichment_cache
        
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

                # Assess hazards
                hazards = self._assess_hazards(detections)

                # 3. Fast Mode Pathway (YOLO -> Fast narration immediately returned)
                if mode == "fast":
                    if is_static or is_similar:
                        logger.info("[CACHE] Reused perception graph and narration")
                        perception_graph = state["cached_perception_graph"]
                        narration = state["cached_narration"]
                        caption = enrichment_cache.get(sid, "florence_caption") or state["cached_caption"] or ""
                    else:
                        # Extract colors fast
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
                                
                        # Update temporal tracker
                        current_time_tracker = asyncio.get_event_loop().time()
                        tracked_results = state["tracker"].update([
                            {"label": d.label, "box": d.box, "confidence": d.confidence}
                            for d in detections
                        ], current_colors, current_time_tracker)
                        
                        # Build perception graph
                        perception_graph = self.fast_narration_service.build_perception_graph(tracked_results, w, h)
                        
                        # Generate fast narration
                        is_first_frame = len(state["memory"]) == 0
                        narration = self.fast_narration_service.generate_narration(
                            perception_graph, state["tracker"], hazards, is_first_frame
                        )
                        
                        # Reuse background enrichment caption if available
                        caption = enrichment_cache.get(sid, "florence_caption") or ""
                        
                        # Launch background task for Florence-2
                        asyncio.create_task(self.run_background_florence(image_bytes, sid))
                        
                        # Update cache variables since it was a successful fresh fast run
                        state["cached_image_gray"] = curr_gray
                        state["cached_detections"] = detections
                        state["cached_perception_graph"] = perception_graph
                        state["cached_caption"] = caption
                        state["cached_narration"] = narration
                        state["cached_hazards"] = hazards
                        state["cached_time"] = current_time_tracker
                
                # 4. Slow/Medium Modes Pathway (Synchronous execution)
                else:
                    if is_static or is_similar:
                        logger.info("[CACHE] Reused Florence caption and narration")
                        perception_graph = state["cached_perception_graph"]
                        narration = state["cached_narration"]
                        caption = state["cached_caption"]
                    else:
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
                                timeout=settings.CAPTION_TIMEOUT_SECONDS
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
                                
                            logger.info(
                                f"[MIGRATION EVAL] BLIP Caption: '{blip_caption}' ({blip_latency:.1f}ms) "
                                f"| Florence-2 Caption: '{florence_caption}'"
                            )
                            
                            if settings.MIGRATION_STAGE == 1:
                                caption = blip_caption
                            else:
                                caption = florence_caption if florence_caption else blip_caption
                                
                        except asyncio.TimeoutError:
                            logger.warning("[FALLBACK] Florence/BLIP captioning timed out. Falling back to fast narration.")
                            florence_timeout_occurred = True
                            caption = ""
                            
                        # Run Florence-2 <OD> task if mode is slow and not timed out
                        florence_objects = {"bboxes": [], "labels": []}
                        if mode == "slow" and not florence_timeout_occurred:
                            logger.info("[SCHEDULER] Slow Loop: Running Florence-2 <OD> task")
                            try:
                                florence_objects = await asyncio.wait_for(
                                    self.florence_service.get_objects(pil_image),
                                    timeout=settings.CAPTION_TIMEOUT_SECONDS
                                )
                            except asyncio.TimeoutError:
                                logger.warning("[FALLBACK] Florence <OD> task timed out. Switched to fast mode.")
                                florence_timeout_occurred = True
                                
                        if florence_timeout_occurred:
                            logger.warning("[FALLBACK] Switched to fast grounded narration after timeout")
                            narration = self.narration_service.generate_fallback_narration(
                                detections=detections,
                                hazards=hazards,
                                pil_image=pil_image
                            )
                            # Fallback fast graph creation
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
                                    
                            current_time_tracker = asyncio.get_event_loop().time()
                            tracked_results = state["tracker"].update([
                                {"label": d.label, "box": d.box, "confidence": d.confidence}
                                for d in detections
                            ], current_colors, current_time_tracker)
                            perception_graph = self.fast_narration_service.build_perception_graph(tracked_results, w, h)
                        else:
                            # Parse fused objects
                            raw_objects = []
                            for d in detections:
                                raw_objects.append({
                                    "label": d.label,
                                    "box": d.box,
                                    "confidence": d.confidence,
                                    "source": "YOLO"
                                })
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
                            
                            fused_objects = self._fuse_detections(raw_objects)
                            
                            current_colors = []
                            for obj in fused_objects:
                                with trace_stage("COLOR_CROP"):
                                    cropped_pil = self.crop_service.crop_object(pil_image, obj["box"])
                                try:
                                    color_res = self.color_service.analyze_color(cropped_pil, obj["confidence"], fast_mode=False)
                                    current_colors.append(color_res["color_name"])
                                except Exception:
                                    current_colors.append("unknown")
                                finally:
                                    cropped_pil.close()
                                    
                            current_time_tracker = asyncio.get_event_loop().time()
                            tracked_results = state["tracker"].update(fused_objects, current_colors, current_time_tracker)
                            perception_graph = self.fast_narration_service.build_perception_graph(tracked_results, w, h)
                            
                            # Normal narration combining caption
                            narration = self.narration_service.generate_narration(
                                caption=caption,
                                detections=detections,
                                hazards=hazards,
                                pil_image=pil_image,
                                recent_memory=list(state["memory"]),
                                perception_graph=perception_graph
                            )
                            
                            # Cache enrichment in background cache
                            enrichment_cache.set(sid, "florence_caption", caption)
                            
                            # Update cache variables
                            state["cached_image_gray"] = curr_gray
                            state["cached_detections"] = detections
                            state["cached_perception_graph"] = perception_graph
                            state["cached_caption"] = caption
                            state["cached_narration"] = narration
                            state["cached_hazards"] = hazards
                            state["cached_time"] = current_time_tracker

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

    async def run_background_florence(self, image_bytes: bytes, session_id: str):
        """Runs Florence captioning asynchronously in the background, storing the enrichment in the cache."""
        from app.services.enrichment_cache import enrichment_cache
        from app.utils.image import load_image_from_bytes
        import time as pytime

        logger.info(f"[TELEMETRY] Starting background Florence-2 caption generation for session: {session_id}")
        start_time = pytime.perf_counter()
        
        try:
            pil_image = load_image_from_bytes(image_bytes)
            try:
                caption = await self.florence_service.get_detailed_caption(pil_image)
                
                # Fetch detailed Florence-2 objects for enrichment
                florence_objects = await self.florence_service.get_objects(pil_image)
                
                enrichment = {
                    "timestamp": pytime.time(),
                    "caption": caption,
                    "florence_objects": florence_objects,
                    "scene_summary": caption
                }
                
                # Store in session memory / enrichment cache
                enrichment_cache.set(session_id, "florence_caption", caption)
                enrichment_cache.set(session_id, "scene_enrichment", enrichment)
                
                # Update session state cache too
                state = self._get_session_state(session_id)
                state["cached_caption"] = caption
                
                elapsed = (pytime.perf_counter() - start_time) * 1000
                logger.info(f"[TELEMETRY] Background Florence-2 completed successfully in {elapsed:.2f}ms for session: {session_id}")
            finally:
                pil_image.close()
        except Exception as e:
            logger.error(f"[TELEMETRY] Background Florence-2 failed for session {session_id}: {str(e)}")

    async def stream_scene_analysis(
        self,
        image_bytes: bytes,
        is_mirrored: bool = False,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Progressively yields scene analysis events (YOLO -> Tracker -> Color -> Florence enrichment)."""
        import time as pytime
        start_time = pytime.perf_counter()
        sid = session_id or "global_default"
        state = self._get_session_state(sid)
        is_first_frame = len(state["memory"]) == 0

        # Load image
        from app.utils.image import load_image_from_bytes
        pil_image = load_image_from_bytes(image_bytes)
        w, h = pil_image.size

        # Stage 1: 0.3s - YOLO detection only
        detect_res = await self.detect_service.detect_objects(pil_image)
        detections = detect_res.detections if detect_res.success else []
        hazards = self._assess_hazards(detections)
        
        temp_tracker = TemporalConsistencyTracker(max_inactive_frames=3, iou_threshold=0.3)
        temp_tracker.update([
            {"label": d.label, "box": d.box, "confidence": d.confidence}
            for d in detections
        ], ["unknown"] * len(detections), pytime.time())
        pg_yolo = self.fast_narration_service.build_perception_graph(
            [{"label": t["label"], "box": t["box"], "confidence": t["confidence"], "color": t["color"]} for t in temp_tracker.update([], [], pytime.time())],
            w, h
        )
        narration_yolo = self.fast_narration_service.generate_narration(pg_yolo, temp_tracker, hazards, is_first_frame=True)
        yield json.dumps({
            "stage": "yolo",
            "time_offset_ms": round((pytime.perf_counter() - start_time) * 1000, 1),
            "narration": narration_yolo
        }) + "\n"

        # Stage 2: 0.5s - Tracker integrated
        current_colors = ["unknown"] * len(detections)
        tracked_results = state["tracker"].update([
            {"label": d.label, "box": d.box, "confidence": d.confidence}
            for d in detections
        ], current_colors, pytime.time())
        pg_tracker = self.fast_narration_service.build_perception_graph(tracked_results, w, h)
        narration_tracker = self.fast_narration_service.generate_narration(pg_tracker, state["tracker"], hazards, is_first_frame)
        yield json.dumps({
            "stage": "tracker",
            "time_offset_ms": round((pytime.perf_counter() - start_time) * 1000, 1),
            "narration": narration_tracker
        }) + "\n"

        # Stage 3: 1.0s - Color classification
        colors = []
        for d in detections:
            cropped = self.crop_service.crop_object(pil_image, d.box)
            try:
                color_res = self.color_service.analyze_color(cropped, d.confidence, fast_mode=True)
                colors.append(color_res["color_name"])
            except Exception:
                colors.append("unknown")
            finally:
                cropped.close()
                
        tracked_results = state["tracker"].update([
            {"label": d.label, "box": d.box, "confidence": d.confidence}
            for d in detections
        ], colors, pytime.time())
        pg_color = self.fast_narration_service.build_perception_graph(tracked_results, w, h)
        narration_color = self.fast_narration_service.generate_narration(pg_color, state["tracker"], hazards, is_first_frame)
        yield json.dumps({
            "stage": "color",
            "time_offset_ms": round((pytime.perf_counter() - start_time) * 1000, 1),
            "narration": narration_color
        }) + "\n"

        # Stage 4: 2.0s - Florence scene enrichment completion
        try:
            caption = await self.florence_service.get_detailed_caption(pil_image)
            from app.services.enrichment_cache import enrichment_cache
            enrichment_cache.set(sid, "florence_caption", caption)
            
            yield json.dumps({
                "stage": "enrichment",
                "time_offset_ms": round((pytime.perf_counter() - start_time) * 1000, 1),
                "narration": "Additional scene details are available.",
                "data": {"caption": caption}
            }) + "\n"
        except Exception as e:
            logger.error(f"Florence stream task failed: {str(e)}")
            
        pil_image.close()


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
