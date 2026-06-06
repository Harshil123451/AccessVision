import re
from typing import List, Dict, Any, Optional, Set
from PIL import Image
from app.services.base import BaseService
from app.schemas.detect import DetectionItem
from app.schemas.scene import HazardItem
from app.schemas.perception import PerceptionGraph
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

    def filter_hallucinations_v2(self, caption: str, detections: List[DetectionItem]) -> str:
        """Aggressively filters out sentences from the caption containing ungrounded entities or unsupported people/objects."""
        detected_labels = {d.label.lower() for d in detections}
        has_person_yolo = any(p in detected_labels for p in ["person", "man", "woman"])
        
        sentences = re.split(r"(?<=[.!?])\s+", caption)
        filtered_sentences = []
        
        person_keywords = {"person", "man", "woman", "people", "lady", "girl", "boy", "someone", "guy", "child", "individual"}
        
        for s in sentences:
            s_lower = s.lower().strip()
            if not s_lower:
                continue
                
            # 1. Aggressive Person check: if YOLO did not see a person, filter out person mentions
            if not has_person_yolo:
                words = set(re.findall(r"\b\w+\b", s_lower))
                if words.intersection(person_keywords):
                    logger.info(f"[GROUNDING] Removed unsupported entity (person) in sentence: '{s}'")
                    continue
            
            # 2. General COCO classes check
            contains_unsupported = False
            for cls in COCO_CLASSES:
                pattern = rf"\b{cls}s?\b"
                if re.search(pattern, s_lower):
                    # Check if this class is grounded in YOLO detections
                    is_grounded = False
                    for dl in detected_labels:
                        if cls in dl or dl in cls:
                            is_grounded = True
                              
                            break
                    if not is_grounded:
                        logger.info(f"[GROUNDING] Removed unsupported entity: '{cls}' in sentence: '{s}'")
                        contains_unsupported = True
                        break
            
            if not contains_unsupported:
                filtered_sentences.append(s)
                
        return " ".join(filtered_sentences)

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
        recent_memory: Optional[List[Dict[str, Any]]] = None,
        perception_graph: Optional[PerceptionGraph] = None
    ) -> str:
        """Assembles a narrative description of the scene focused purely on grounded object perception."""
        logger.info("Generating accessibility-focused narration")
        from app.core.telemetry import trace_stage
        
        with trace_stage("NARRATION"):
            # 1. Noun Validation Layer (Hallucination Suppression V2)
            clean_caption = self.filter_hallucinations_v2(caption, detections)
            
            # 2. Critical Hazards first
            hazard_warnings = []
            for h in hazards:
                if h.severity in ["medium", "high"]:
                    hazard_warnings.append(h.description)
            warnings_str = " ".join(hazard_warnings)

            # 3. Handle rolling memory contradiction checks
            contradiction_warnings = []
            if recent_memory:
                last_frame = recent_memory[-1]
                prev_labels = {d["label"].lower() for d in last_frame.get("detections", [])}
                curr_labels = {d.label.lower() for d in detections}
                for pl in prev_labels:
                    if pl not in curr_labels:
                        contradiction_warnings.append(f"I am no longer confident that a {pl} is present.")
            
            # 4. Describe grounded objects object-by-object from the Perception Graph
            object_descriptions = []
            grounded_labels = []
            has_low_confidence_detections = False
            
            if perception_graph and perception_graph.objects:
                for obj in perception_graph.objects:
                    # Skip describing if it is a high-severity hazard to prevent repetition
                    is_hazard = any(h.label.lower() == obj.class_name.lower() for h in hazards if h.severity in ["medium", "high"])
                    if is_hazard:
                        continue

                    grounded_labels.append(obj.class_name)
                    color_phrase = f"{obj.color} " if obj.color != "unknown" else ""
                    
                    # Sentence formatting based on confidence tier
                    if obj.narration_confidence == "HIGH":
                        # E.g., "A blue suitcase is in the foreground to your left."
                        a_an = "An" if obj.class_name[0].lower() in 'aeiou' and not color_phrase else "A"
                        if color_phrase:
                            a_an = "An" if color_phrase[0].lower() in 'aeiou' else "A"
                        object_descriptions.append(f"{a_an} {color_phrase}{obj.class_name} is {obj.position}.")
                    elif obj.narration_confidence == "MEDIUM":
                        # E.g., "There appears to be a blue suitcase nearby in the center."
                        a_an = "an" if obj.class_name[0].lower() in 'aeiou' and not color_phrase else "a"
                        if color_phrase:
                            a_an = "an" if color_phrase[0].lower() in 'aeiou' else "a"
                        object_descriptions.append(f"There appears to be {a_an} {color_phrase}{obj.class_name} {obj.position}.")
                    else:
                        has_low_confidence_detections = True
            
            # 5. Build final narration
            parts = []
            if warnings_str:
                parts.append(f"Caution: {warnings_str}")

            scene_type = self.get_scene_type(clean_caption)
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
                
            logger.info(f"[SCENE] Final grounded entities: {', '.join(grounded_labels)}")
            logger.info(f"[SCENE] Narration confidence: {overall_confidence}")

            return " ".join(parts)

    def generate_fallback_narration(
        self, 
        detections: List[DetectionItem], 
        hazards: List[HazardItem],
        pil_image: Optional[Image.Image] = None
    ) -> str:
        """Generates a complete, intentional accessibility narration when Florence-2 times out."""
        if not detections:
            return "I cannot currently detect any objects. No immediate obstacles are detected nearby."

        # Get image dimensions
        w, h = pil_image.size if pil_image else (640, 640)

        object_sentences = []
        for i, d in enumerate(detections):
            # Focus on the top 4 primary detections to keep narration concise and clear
            if i >= 4:
                break

            label = d.label.lower()
            xmin, ymin, xmax, ymax = d.box
            area = ((xmax - xmin) * (ymax - ymin)) / (w * h)
            center_x = (xmin + xmax) / 2.0 / w
            ymax_norm = ymax / h

            # Size classification
            if area >= 0.20:
                size_desc = "large"
            elif area >= 0.05:
                size_desc = "medium"
            else:
                size_desc = "small"

            # Spatial mapping
            x_pos = "left" if center_x < 0.33 else ("right" if center_x > 0.66 else "center")
            if x_pos == "center":
                if ymax_norm > 0.5:
                    position_phrase = "directly ahead of you"
                else:
                    position_phrase = "in the background center"
            else:
                position_phrase = f"to your {x_pos}"

            # Fast path color classification
            color_name = "unknown"
            if pil_image:
                from app.core.telemetry import trace_stage
                with trace_stage("COLOR_CROP"):
                    cropped_pil = self.crop_service.crop_object(pil_image, d.box)
                try:
                    color_res = self.color_service.analyze_color(cropped_pil, d.confidence, fast_mode=True)
                    color_name = color_res["color_name"]
                except Exception:
                    pass
                finally:
                    cropped_pil.close()

            # Construct sentence structures
            a_an = "An" if label[0].lower() in 'aeiou' else "A"
            if size_desc == "medium":
                object_sentences.append(f"{a_an} {label} is {position_phrase}.")
            else:
                object_sentences.append(f"{a_an} {size_desc} {label} is {position_phrase}.")

            if color_name != "unknown":
                object_sentences.append(f"The {label} appears {color_name} in colour.")

        # Obstacle analysis
        hazard_warnings = []
        for hz in hazards:
            if hz.severity in ["medium", "high"]:
                hazard_warnings.append(hz.description)

        if hazard_warnings:
            warnings_str = " ".join(hazard_warnings)
            object_sentences.append(f"Caution: {warnings_str}")
        else:
            object_sentences.append("No immediate obstacles are detected nearby.")

        return " ".join(object_sentences)
