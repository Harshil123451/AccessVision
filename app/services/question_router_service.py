import re
from typing import Tuple, Optional, List
from PIL import Image
from app.services.base import BaseService
from app.services.vqa_service import VqaService
from app.services.detect_service import DetectService
from app.services.crop_service import CropService
from app.services.color_service import ColorService
from app.services.scene_service import SceneService
from app.schemas.reasoning import ReasoningResult
from app.schemas.detect import DetectionItem
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

    def analyze_question(self, question: str) -> Tuple[str, Optional[str]]:
        """Parses the question to identify intent type and any referenced COCO objects."""
        q_lower = question.lower().strip()

        # 1. Detect target object (look for COCO classes in the question)
        target_object = None
        # Sort classes by length descending to match longer multi-word classes first
        sorted_classes = sorted(COCO_CLASSES, key=len, reverse=True)
        for cls in sorted_classes:
            # Use word boundaries to avoid partial matches (e.g., "car" in "carpet")
            pattern = rf"\b{cls}s?\b"
            if re.search(pattern, q_lower):
                target_object = cls
                break

        # 2. Detect intent type based on trigger keywords
        if "color" in q_lower:
            intent = "color"
        elif any(kw in q_lower for kw in ["how many", "count", "number of"]):
            intent = "counting"
        elif any(kw in q_lower for kw in ["is there", "are there", "do you see", "presence"]):
            intent = "object_presence"
        elif any(kw in q_lower for kw in ["read", "text", "sign", "writing", "ocr", "says"]):
            intent = "reading_text"
        elif any(kw in q_lower for kw in ["describe", "caption", "what do you see", "look like"]):
            intent = "scene_description"
        elif any(kw in q_lower for kw in ["next to", "above", "below", "left of", "right of", "beside", "near"]):
            intent = "spatial_reasoning"
        else:
            intent = "generic_vqa"

        logger.info(f"Routed question: '{question}' -> intent: {intent}, target_object: {target_object}")
        return intent, target_object

    async def route_and_reason(self, image_bytes: bytes, question: str, is_mirrored: bool = False) -> ReasoningResult:
        """Orchestrates the grounded multimodal reasoning flow."""
        intent, target_object = self.analyze_question(question)
        from app.core.telemetry import trace_stage, get_current_telemetry
        import time
        
        telemetry = get_current_telemetry()
        start_time = time.perf_counter()
        
        # Load and parse original image
        with trace_stage("IMAGE_DECODE"):
            pil_image = load_image_from_bytes(image_bytes)
        
        detections: List[DetectionItem] = []
        answer = ""
        grounded_by = "fallback"
        
        try:
            with trace_stage("ORCHESTRATOR"):
                
                # --- PATHWAY 1: COLOR ANALYSIS ---
                if intent == "color":
                    # 1. Detect objects first
                    detect_res = await self.detect_service.detect_objects(image_bytes)
                    detections = detect_res.detections
                    
                    # 2. Find target object detections
                    matching_dets = [d for d in detections if d.label == target_object] if target_object else []
                    
                    if matching_dets:
                        # Select the highest confidence detection
                        best_match = max(matching_dets, key=lambda x: x.confidence)
                        # Crop image region
                        cropped_pil = self.crop_service.crop_object(pil_image, best_match.box)
                        try:
                            # Extract dominant color directly without VQA
                            color = self.color_service.get_dominant_color(cropped_pil)
                        finally:
                            cropped_pil.close()
                        if best_match.confidence >= 0.75:
                            answer = f"The {target_object} is {color}."
                        else:
                            answer = f"I think the {target_object} may be {color}."
                        grounded_by = "color_service"
                    else:
                        # Grounded Uncertainty Response: avoid color hallucination for non-existent objects
                        if target_object:
                            answer = f"I could not confidently detect any {target_object} in the image to determine its color."
                            grounded_by = "grounding_uncertainty"
                        else:
                            logger.info("No target object identified for color query. Falling back to general VQA.")
                            vqa_res = await self.vqa_service.answer_question(image_bytes, question)
                            answer = vqa_res.answer
                            grounded_by = "vqa_fallback"

                # --- PATHWAY 2: COUNTING ---
                elif intent == "counting":
                    detect_res = await self.detect_service.detect_objects(image_bytes)
                    detections = detect_res.detections
                    
                    if target_object:
                        matching_dets = [d for d in detections if d.label == target_object]
                        count = len(matching_dets)
                        plural = target_object + "s" if not target_object.endswith("s") else target_object
                        if count == 0:
                            answer = f"I could not confidently detect any {plural} in the image."
                        else:
                            all_high_conf = all(d.confidence >= 0.75 for d in matching_dets)
                            if all_high_conf:
                                if count == 1:
                                    answer = f"There is one {target_object} in the image."
                                else:
                                    answer = f"There are {count} {plural} in the image."
                            else:
                                if count == 1:
                                    answer = f"I think there may be a {target_object} in the image."
                                else:
                                    answer = f"I think there may be about {count} {plural} in the image."
                    else:
                        # Generic counting: count total objects
                        count = len(detections)
                        if count == 0:
                            answer = "I do not see any objects in the scene."
                        else:
                            all_high_conf = all(d.confidence >= 0.75 for d in detections)
                            if all_high_conf:
                                if count == 1:
                                    answer = "I see only one object in the scene."
                                else:
                                    answer = f"I count {count} objects in the scene."
                            else:
                                if count == 1:
                                    answer = "I think I see one object in the scene."
                                else:
                                    answer = f"I think there may be about {count} objects in the scene."
                    grounded_by = "yolo_detection"

                # --- PATHWAY 3: OBJECT PRESENCE ---
                elif intent == "object_presence":
                    detect_res = await self.detect_service.detect_objects(image_bytes)
                    detections = detect_res.detections
                    
                    if target_object:
                        matching_dets = [d for d in detections if d.label == target_object]
                        plural = target_object + "s" if not target_object.endswith("s") else target_object
                        if matching_dets:
                            confidence = max(d.confidence for d in matching_dets)
                            if confidence >= 0.75:
                                answer = f"Yes, I confidently detected a {target_object} in the scene (with {confidence:.0%} confidence)."
                            else:
                                answer = f"I think there may be a {target_object} in the scene (detected with {confidence:.0%} confidence)."
                        else:
                            answer = f"No, I did not detect any {plural} in the scene."
                        grounded_by = "yolo_detection"
                    else:
                        # Fallback to VQA if no clear target object identified
                        vqa_res = await self.vqa_service.answer_question(image_bytes, question)
                        answer = vqa_res.answer
                        grounded_by = "vqa_fallback"

                # --- PATHWAY 4: READING TEXT (OCR) ---
                elif intent == "reading_text":
                    detect_res = await self.detect_service.detect_objects(image_bytes)
                    detections = detect_res.detections
                    
                    text_bearing_classes = {"stop sign", "book", "bottle", "tv", "laptop", "cell phone"}
                    text_dets = [d for d in detections if d.label in text_bearing_classes]
                    
                    if text_dets:
                        # Crop text bearing region to ground VQA
                        best_match = max(text_dets, key=lambda x: x.confidence)
                        cropped_pil = self.crop_service.crop_object(pil_image, best_match.box)
                        try:
                            cropped_bytes = save_image_to_bytes(cropped_pil)
                        finally:
                            cropped_pil.close()
                        
                        vqa_res = await self.vqa_service.answer_question(cropped_bytes, question)
                        if best_match.confidence >= 0.75:
                            answer = f"Based on the isolated {best_match.label} region: {vqa_res.answer}"
                        else:
                            answer = f"I think I see a {best_match.label} region. Based on that: {vqa_res.answer}"
                        grounded_by = "cropped_vqa_ocr"
                    else:
                        # Fallback to full-image VQA with disclaimer
                        vqa_res = await self.vqa_service.answer_question(image_bytes, question)
                        if target_object:
                            answer = f"I did not detect a clear {target_object} in the image, but reading the overall scene: {vqa_res.answer}"
                        else:
                            answer = f"I did not detect any clear sign or text-bearing objects. Here is a general reading: {vqa_res.answer}"
                        grounded_by = "vqa_fallback"

                # --- PATHWAY 5: SCENE DESCRIPTION ---
                elif intent == "scene_description":
                    scene_res = await self.scene_service.analyze_scene(image_bytes, is_mirrored=is_mirrored)
                    answer = scene_res.narration
                    detections = scene_res.objects
                    grounded_by = "scene_service"

                # --- PATHWAY 6: SPATIAL REASONING ---
                elif intent == "spatial_reasoning":
                    # Run detection to locate object positions for context
                    detect_res = await self.detect_service.detect_objects(image_bytes)
                    detections = detect_res.detections
                    
                    # Format a context string listing detected objects and their boxes to help ground spatial reasoning
                    if detections:
                        context_elements = []
                        for d in detections:
                            # Box is [xmin, ymin, xmax, ymax]
                            xmin, ymin, xmax, ymax = d.box
                            # Describe spatial coordinates in human readable terms
                            w, h = pil_image.size
                            center_x = (xmin + xmax) / 2.0 / w
                            center_y = (ymin + ymax) / 2.0 / h
                            
                            x_pos = "left" if center_x < 0.33 else ("right" if center_x > 0.66 else "center")
                            y_pos = "top" if center_y < 0.33 else ("bottom" if center_y > 0.66 else "middle")
                            
                            context_elements.append(f"{d.label} at the {y_pos}-{x_pos} of the image")
                        
                        grounding_context = "Grounded visual object coordinates: " + ", ".join(context_elements) + ". "
                        augmented_question = f"{grounding_context}Question: {question}"
                    else:
                        augmented_question = question
                    
                    vqa_res = await self.vqa_service.answer_question(image_bytes, augmented_question)
                    answer = vqa_res.answer
                    grounded_by = "grounded_vqa_spatial"

                # --- PATHWAY 7: GENERIC VQA (FALLBACK) ---
                else:
                    vqa_res = await self.vqa_service.answer_question(image_bytes, question)
                    answer = vqa_res.answer
                    grounded_by = "vqa_fallback"
        finally:
            pil_image.close()
            
        # Log telemetry details if not already handled by scene service
        if intent != "scene_description":
            has_uncertainty = any(d.confidence < 0.75 for d in detections) if detections else False
            narration_confidence = "LOW" if has_uncertainty else "HIGH"
            logger.info(f"[SCENE] Narration confidence: {narration_confidence}")
            if is_mirrored:
                logger.info("[CAMERA] Mirrored preview enabled")
            else:
                logger.info("[CAMERA] Mirrored preview disabled")

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
