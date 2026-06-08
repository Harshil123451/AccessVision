import re
import asyncio
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image
from app.services.base import BaseService
from app.services.vqa_service import VqaService
from app.services.detect_service import DetectService
from app.services.crop_service import CropService
from app.services.color_service import ColorService
from app.services.scene_service import SceneService
from app.services.florence_service import FlorenceService
from app.services.enrichment_cache import enrichment_cache
from app.core.config import settings
from app.schemas.reasoning import ReasoningResult
from app.schemas.detect import DetectionItem
from app.schemas.perception import PerceptionGraph, GroundedObject
from app.utils.image import load_image_from_bytes, save_image_to_bytes
import logging

logger = logging.getLogger("accessvision")

# 80 standard COCO classes
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]

class QuestionRouterService(BaseService):
    """Lightweight intent router and grounded orchestration service.
    
    Inspects user questions, detects the semantic intent and target objects,
    and routes the request to dedicated visual reasoning pathways (crops,
    color extraction, spatial rules) instead of defaulting blindly to VQA.
    """

    def __init__(self):
        self.vqa_service = VqaService()
        self.detect_service = DetectService()
        self.crop_service = CropService()
        self.color_service = ColorService()
        self.scene_service = SceneService()
        self.florence_service = FlorenceService()

    def analyze_question(self, question: str) -> Tuple[str, Optional[str]]:
        """Parses the question to identify intent type and any referenced COCO objects."""
        q_lower = question.lower().strip()

        # 1. Detect target object (look for COCO classes in the question)
        target_object = None
        sorted_classes = sorted(COCO_CLASSES, key=len, reverse=True)
        for cls in sorted_classes:
            pattern = rf"\b{cls}s?\b"
            if re.search(pattern, q_lower):
                target_object = cls
                break

        # 2. Classify intent type into Phase 3 classes
        # FAST_COLOR_QUERY
        if "color" in q_lower:
            intent = "FAST_COLOR_QUERY"
        # OCR_QUERY
        elif any(kw in q_lower for kw in ["read", "label", "text", "sign", "writing", "ocr", "says", "words"]):
            intent = "OCR_QUERY"
        # DETAILED_SCENE_QUERY
        elif any(kw in q_lower for kw in ["describe the room", "describe the scene", "describe the environment", "what do you see"]):
            intent = "DETAILED_SCENE_QUERY"
        # FAST_OBJECT_QUERY keywords
        elif any(kw in q_lower for kw in ["in front of", "ahead", "blocking", "obstacle", "where is", "is there", "are there", "do you see", "how many", "count"]):
            intent = "FAST_OBJECT_QUERY"
        elif "what objects" in q_lower or "what is in front" in q_lower or "what do i have in front" in q_lower:
            intent = "FAST_OBJECT_QUERY"
        elif "what do you see" in q_lower:
            intent = "DETAILED_SCENE_QUERY"
        else:
            # Fallback check
            if target_object and any(q_lower.startswith(prefix) for prefix in ["is there", "are there", "do you see", "can you see"]):
                intent = "FAST_OBJECT_QUERY"
            else:
                intent = "VQA_QUERY"

        logger.info(f"Routed question: '{question}' -> intent: {intent}, target_object: {target_object}")
        return intent, target_object

    def classify_query_speed(self, question: str) -> str:
        """Classifies incoming natural language questions into fast, medium, or slow mode."""
        intent, _ = self.analyze_question(question)
        if intent in ["FAST_OBJECT_QUERY", "FAST_COLOR_QUERY"]:
            return "fast"
        elif intent == "DETAILED_SCENE_QUERY":
            return "medium"
        else:
            return "slow"

    async def route_and_reason(self, image_bytes: bytes, question: str, is_mirrored: bool = False, session_id: Optional[str] = None) -> ReasoningResult:
        """Orchestrates the grounded multimodal reasoning flow."""
        intent, target_object = self.analyze_question(question)
        mode = self.classify_query_speed(question)
        logger.info(f"[ROUTER] Query classified as {mode.upper()} (Intent: {intent})")
        
        from app.core.telemetry import trace_stage, get_current_telemetry
        import time
        
        telemetry = get_current_telemetry()
        start_time = time.perf_counter()
        
        # Load and parse original image
        with trace_stage("IMAGE_DECODE"):
            pil_image = load_image_from_bytes(image_bytes)
        w, h = pil_image.size
        
        detections: List[DetectionItem] = []
        answer = ""
        grounded_by = "fallback"
        sid = session_id or "global_default"
        
        try:
            with trace_stage("ORCHESTRATOR"):
                
                # --- PATHWAY 1: FAST OBJECT QUERY (YOLO + Perception Graph only) ---
                if intent == "FAST_OBJECT_QUERY":
                    # Run YOLO
                    detect_res = await self.detect_service.detect_objects(pil_image)
                    detections = detect_res.detections if detect_res.success else []
                    
                    # Track colors fast
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
                            
                    # Update tracker
                    state = self.scene_service._get_session_state(sid)
                    tracked_results = state["tracker"].update([
                        {"label": d.label, "box": d.box, "confidence": d.confidence}
                        for d in detections
                    ], current_colors, time.time())
                    
                    # Build graph
                    pg = self.scene_service.fast_narration_service.build_perception_graph(tracked_results, w, h)
                    
                    # Formulate answer from Perception Graph
                    if "is there" in question.lower() or "are there" in question.lower() or "do you see" in question.lower():
                        if target_object:
                            matching_obj = next((o for o in pg.objects if o.class_name.lower() == target_object.lower()), None)
                            if matching_obj:
                                answer = f"Yes, a {target_object} is {matching_obj.position}."
                            else:
                                answer = f"No, I do not see a {target_object} in front of you."
                        else:
                            if pg.objects:
                                answer = f"Yes, I see {len(pg.objects)} objects."
                            else:
                                answer = "No, I do not see any objects in front of you."
                    elif "how many" in question.lower() or "count" in question.lower():
                        if target_object:
                            matching_count = sum(1 for o in pg.objects if o.class_name.lower() == target_object.lower())
                            plural = target_object + "s" if not target_object.endswith("s") else target_object
                            if matching_count == 1:
                                answer = f"There is one {target_object}."
                            elif matching_count > 1:
                                answer = f"There are {matching_count} {plural}."
                            else:
                                answer = f"I could not detect any {plural}."
                        else:
                            if len(pg.objects) == 1:
                                answer = "I see only one object."
                            elif len(pg.objects) > 1:
                                answer = f"I count {len(pg.objects)} objects."
                            else:
                                answer = "I do not see any objects."
                    else:
                        # Describe what objects are in front of user
                        if pg.objects:
                            desc_list = []
                            for o in pg.objects:
                                color_prefix = f"a {o.color}" if o.color != "unknown" else "a"
                                if color_prefix == "a" and o.class_name[0].lower() in 'aeiou':
                                    color_prefix = "an"
                                desc_list.append(f"{color_prefix} {o.class_name} {o.position}")
                            answer = "I see: " + ", ".join(desc_list) + "."
                        else:
                            answer = "I do not see any objects in front of you."
                            
                    grounded_by = "yolo_perception_graph"

                # --- PATHWAY 2: FAST COLOR QUERY (YOLO + Color Service) ---
                elif intent == "FAST_COLOR_QUERY":
                    # Run YOLO
                    detect_res = await self.detect_service.detect_objects(pil_image)
                    detections = detect_res.detections if detect_res.success else []
                    
                    # Find target object detections
                    matching_dets = [d for d in detections if d.label == target_object] if target_object else []
                    
                    if matching_dets:
                        best_match = max(matching_dets, key=lambda x: x.confidence)
                        with trace_stage("COLOR_CROP"):
                            cropped_pil = self.crop_service.crop_object(pil_image, best_match.box)
                        try:
                            color_analysis = self.color_service.analyze_color(
                                cropped_pil, 
                                best_match.confidence, 
                                fast_mode=True
                            )
                            color_name = color_analysis["color_name"]
                            color_conf = color_analysis["confidence"]
                        finally:
                            cropped_pil.close()
                        
                        if color_conf == "HIGH":
                            answer = f"The {target_object} is {color_name}."
                        elif color_conf == "MEDIUM":
                            answer = f"The {target_object} appears to be {color_name}."
                        else: # LOW
                            answer = f"I cannot confidently determine the color of the {target_object}."
                        grounded_by = "yolo_color_service"
                    else:
                        if target_object:
                            answer = f"I could not confidently detect any {target_object} in the image to determine its color."
                            grounded_by = "grounding_uncertainty"
                        else:
                            answer = "Please specify what object color you want to know."
                            grounded_by = "routing_fallback"

                # --- PATHWAY 3: OCR QUERY (Florence OCR) ---
                elif intent == "OCR_QUERY":
                    # Check background enrichment cache first
                    cached_ocr = enrichment_cache.get(sid, "ocr_result")
                    if cached_ocr:
                        answer = f"The text detected in the image reads: '{cached_ocr}'"
                        grounded_by = "enrichment_cache"
                    else:
                        ocr_text = await self.florence_service.get_ocr(image_bytes)
                        if ocr_text:
                            answer = f"The text detected in the image reads: '{ocr_text}'"
                            enrichment_cache.set(sid, "ocr_result", ocr_text)
                        else:
                            answer = "No text was detected in the scene."
                        grounded_by = "florence_ocr"

                # --- PATHWAY 4: DETAILED SCENE QUERY (Florence Captioning) ---
                elif intent == "DETAILED_SCENE_QUERY":
                    # Check background enrichment cache first
                    cached_caption = enrichment_cache.get(sid, "florence_caption")
                    if cached_caption:
                        answer = cached_caption
                        grounded_by = "enrichment_cache"
                    else:
                        caption = await self.florence_service.get_detailed_caption(pil_image)
                        answer = caption
                        enrichment_cache.set(sid, "florence_caption", caption)
                        grounded_by = "florence_captioning"

                # --- PATHWAY 5: VQA QUERY (Florence VQA) ---
                elif intent == "VQA_QUERY":
                    cache_key = f"vqa_{question.lower().strip()}"
                    cached_vqa = enrichment_cache.get(sid, cache_key)
                    if cached_vqa:
                        answer = cached_vqa
                        grounded_by = "enrichment_cache"
                    else:
                        answer_dict = await self.florence_service.run_task(image_bytes, "<VQA>", text_input=question)
                        answer = answer_dict.get("<VQA>", "I could not answer the question based on the image.")
                        enrichment_cache.set(sid, cache_key, answer)
                        grounded_by = "florence_vqa"

        finally:
            pil_image.close()

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return ReasoningResult(
            success=True,
            question=question,
            intent=intent,
            target_object=target_object,
            answer=answer,
            grounded_by=grounded_by,
            detections=detections,
            metrics={"inference_ms": elapsed_ms}
        )

