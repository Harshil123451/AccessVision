import re
from typing import List, Dict, Any, Optional, Set
from PIL import Image
from app.services.base import BaseService
from app.schemas.detect import DetectionItem
from app.schemas.scene import HazardItem
import logging

logger = logging.getLogger("accessvision")

# 80 standard COCO classes plus common synonyms
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
    "toothbrush",
    # Common synonyms/variants to prevent hallucination of related terms
    "table", "sofa", "desk", "computer", "phone", "television", "light"
]

class NarrationService(BaseService):
    """Combines grounded detections, spatial positions, and color details to construct truthful, uncertainty-aware narration."""

    def __init__(self):
        from app.services.crop_service import CropService
        from app.services.color_service import ColorService
        self.crop_service = CropService()
        self.color_service = ColorService()

    def filter_hallucinations(self, caption: str, detections: List[DetectionItem]) -> Set[str]:
        """Validates nouns in the caption against YOLO detections, logging any ungrounded entities."""
        detected_labels = {d.label.lower() for d in detections}
        grounded_entities = set()
        
        # Cross-reference caption words with COCO classes/synonyms
        for cls in COCO_CLASSES:
            pattern = rf"\b{cls}s?\b"
            if re.search(pattern, caption.lower()):
                # Check if this class or a related substring is in the detected labels
                is_grounded = False
                matched_label = None
                for dl in detected_labels:
                    if cls in dl or dl in cls:
                        is_grounded = True
                        matched_label = dl
                        break
                
                if is_grounded:
                    grounded_entities.add(matched_label)
                else:
                    logger.info(f"[GROUNDING] Removed unsupported entity: {cls}")
                    
        return grounded_entities

    def get_scene_type(self, caption: str) -> Optional[str]:
        """Extracts the general scene context (e.g. bedroom, kitchen) from the caption."""
        scene_types = [
            "bedroom", "kitchen", "living room", "office", "classroom", 
            "bathroom", "street", "park", "sidewalk", "store", "restaurant", 
            "hallway", "room", "beach", "field", "road"
        ]
        for s in scene_types:
            if re.search(rf"\b{s}s?\b", caption.lower()):
                return s
        return None

    def generate_narration(
        self, 
        caption: str, 
        detections: List[DetectionItem], 
        hazards: List[HazardItem],
        pil_image: Optional[Image.Image] = None,
        recent_memory: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Assembles a narrative description of the scene focused purely on grounded object perception."""
        logger.info("Generating accessibility-focused narration")
        from app.core.telemetry import trace_stage
        
        with trace_stage("NARRATION"):
            # 1. Noun Validation Layer
            self.filter_hallucinations(caption, detections)
            
            # 2. Critical Hazards first
            hazard_warnings = []
            for h in hazards:
                if h.severity in ["medium", "high"]:
                    hazard_warnings.append(h.description)
            warnings_str = " ".join(hazard_warnings)

            # 3. Handle rolling memory contradiction checks (Phase 4)
            contradiction_warnings = []
            if recent_memory:
                last_frame = recent_memory[-1]
                prev_labels = {d["label"].lower() for d in last_frame.get("detections", [])}
                curr_labels = {d.label.lower() for d in detections}
                for pl in prev_labels:
                    if pl not in curr_labels:
                        contradiction_warnings.append(f"I am no longer confident that a {pl} is present.")
            
            # 4. Describe grounded objects object-by-object (Phase 3 & 5)
            object_descriptions = []
            grounded_labels = []
            has_low_confidence_detections = False
            
            w, h = pil_image.size if pil_image else (640, 640)
            
            for d in detections:
                # Avoid repeating hazards
                is_hazard = any(h.label == d.label for h in hazards if h.severity in ["medium", "high"])
                if is_hazard:
                    continue

                grounded_labels.append(d.label)
                
                # Analyze color
                color_name = "unknown"
                color_conf = "LOW"
                if pil_image:
                    # Crop with safety inset
                    cropped_pil = self.crop_service.crop_object(pil_image, d.box)
                    try:
                        color_res = self.color_service.analyze_color(cropped_pil, d.confidence)
                        color_name = color_res["color_name"]
                        color_conf = color_res["confidence"]
                    finally:
                        cropped_pil.close()
                
                # Determine object spatial layout
                xmin, ymin, xmax, ymax = d.box
                center_x = (xmin + xmax) / 2.0 / w
                center_y = (ymin + ymax) / 2.0 / h
                area = ((xmax - xmin) * (ymax - ymin)) / (w * h)
                
                # Horizontal Mapping
                x_pos = "left" if center_x < 0.33 else ("right" if center_x > 0.66 else "center")
                
                # Depth Mapping (avoiding confusing vertical words)
                is_normalized = all(0.0 <= c <= 1.0 for c in d.box)
                ymax_norm = ymax if is_normalized else ymax / h
                if area >= 0.20:
                    depth = "foreground"
                elif ymax_norm > 0.65:
                    depth = "foreground"
                elif area < 0.04:
                    depth = "background"
                else:
                    depth = "nearby"
                    
                # Combine spatial description
                if x_pos == "center":
                    if depth == "foreground":
                        spatial_phrase = "in the center foreground"
                    elif depth == "background":
                        spatial_phrase = "in the center background"
                    else:
                        spatial_phrase = "nearby in the center"
                else:
                    if depth == "foreground":
                        spatial_phrase = f"in the foreground to your {x_pos}"
                    elif depth == "background":
                        spatial_phrase = f"in the background to your {x_pos}"
                    else:
                        spatial_phrase = f"to your {x_pos}"
                
                # Determine individual confidence tier
                if d.confidence >= 0.75 and color_conf == "HIGH":
                    tier = "HIGH"
                elif d.confidence >= 0.50 and color_conf != "LOW":
                    tier = "MEDIUM"
                else:
                    tier = "LOW"
                
                # Format description based on confidence tier
                a_color = f" {color_name}" if color_name != "unknown" else ""
                
                if tier == "HIGH":
                    object_descriptions.append(f"There is a{a_color} {d.label} {spatial_phrase}.")
                elif tier == "MEDIUM":
                    object_descriptions.append(f"There appears to be a{a_color} {d.label} {spatial_phrase}.")
                else:
                    has_low_confidence_detections = True
            
            # 5. Build final narration
            parts = []
            if warnings_str:
                parts.append(f"Caution: {warnings_str}")

            scene_type = self.get_scene_type(caption)
            if scene_type:
                parts.append(f"The scene appears to be a {scene_type}.")
            else:
                parts.append("The scene shows an environment.")
                
            if object_descriptions:
                parts.extend(object_descriptions)
            
            # Append low-confidence disclaimer
            if has_low_confidence_detections or not detections:
                parts.append("I cannot confidently identify additional objects.")
                
            # Append contradiction warnings
            if contradiction_warnings:
                parts.extend(contradiction_warnings)
                
            # Compute overall narration confidence
            if not detections:
                overall_confidence = "LOW"
            elif any(d.confidence < 0.75 for d in detections) or has_low_confidence_detections:
                overall_confidence = "MEDIUM" if any(d.confidence >= 0.50 for d in detections) else "LOW"
            else:
                overall_confidence = "HIGH"
                
            # Telemetry logging output
            logger.info(f"[SCENE] Final grounded entities: {', '.join(grounded_labels)}")
            logger.info(f"[SCENE] Narration confidence: {overall_confidence}")

            return " ".join(parts)
