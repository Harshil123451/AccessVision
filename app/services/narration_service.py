import re
from typing import List
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
    """Combines detections, scene captioning, and hazards to generate descriptive and hazard-aware narration."""

    def filter_hallucinations(self, caption: str, detections: List[DetectionItem]) -> str:
        """Filters out hallucinated objects from the caption based on YOLO detections."""
        detected_labels = {d.label.lower() for d in detections}
        
        # Identify which COCO classes mentioned in the caption are NOT detected
        hallucinated = []
        for cls in COCO_CLASSES:
            # Check if the class name is present in the caption
            pattern = rf"\b{cls}s?\b"
            if re.search(pattern, caption.lower()):
                if cls not in detected_labels:
                    # Special case: check if it's a synonym or compound noun that might match
                    is_substring = False
                    for dl in detected_labels:
                        if cls in dl or dl in cls:
                            is_substring = True
                            break
                    if not is_substring:
                        hallucinated.append(cls)
                    
        if not hallucinated:
            return caption
            
        cleaned_caption = caption
        for word in hallucinated:
            logger.info(f"[SCENE] Removed hallucinated object: {word}")
            
            is_furniture = word in {"bed", "couch", "chair", "dining table", "table", "sofa", "desk"}
            replacement = "other furniture" if is_furniture else "other items"
            
            # Replace "and a/an/the/some <word>" or "and <word>s" with "and other items"
            pattern_and = rf"\band\s+(?:a\s+|an\s+|the\s+|some\s+)?{word}s?\b"
            cleaned_caption, count = re.subn(pattern_and, f"and {replacement}", cleaned_caption, flags=re.IGNORECASE)
            
            if count == 0:
                # Replace "with a/an/the/some <word>s?"
                pattern_with = rf"\bwith\s+(?:a\s+|an\s+|the\s+|some\s+)?{word}s?\b"
                cleaned_caption, count = re.subn(pattern_with, f"with {replacement}", cleaned_caption, flags=re.IGNORECASE)
                
            if count == 0:
                # General replacement
                pattern_obj = rf"\b(?:a\s+|an\s+|the\s+|some\s+)?{word}s?\b"
                cleaned_caption = re.sub(pattern_obj, replacement, cleaned_caption, flags=re.IGNORECASE)
                
        # Clean double spaces/punctuation
        cleaned_caption = re.sub(r',\s*,\s*', ', ', cleaned_caption)
        cleaned_caption = re.sub(r'\s+', ' ', cleaned_caption).strip()
        return cleaned_caption

    def generate_narration(
        self, 
        caption: str, 
        detections: List[DetectionItem], 
        hazards: List[HazardItem]
    ) -> str:
        """Assembles a narrative description of the scene for visually impaired users.
        
        It handles layout positioning (left, center, right) and warns about hazards first.
        """
        logger.info("Generating accessibility-focused narration")
        from app.core.telemetry import trace_stage
        
        with trace_stage("NARRATION"):
            # Filter hallucinated objects from the BLIP caption using YOLO detections
            filtered_caption = self.filter_hallucinations(caption, detections)
            
            # 1. Handle critical hazards first to warn the user immediately
            hazard_warnings = []
            for h in hazards:
                if h.severity in ["medium", "high"]:
                    hazard_warnings.append(h.description)
                    
            warnings_str = " ".join(hazard_warnings)

            # 2. Group non-hazardous or general objects by location
            position_groups = {}
            
            def get_spatial_position(box) -> str:
                # box is [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box
                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                
                is_normalized = all(0.0 <= c <= 1.0 for c in box)
                w_limit_left = 0.33 if is_normalized else 640 * 0.33
                w_limit_right = 0.66 if is_normalized else 640 * 0.66
                h_limit_upper = 0.33 if is_normalized else 640 * 0.33
                h_limit_lower = 0.66 if is_normalized else 640 * 0.66
                
                # Horizontal position
                if center_x < w_limit_left:
                    x_pos = "left"
                elif center_x > w_limit_right:
                    x_pos = "right"
                else:
                    x_pos = "center"
                    
                # Vertical position
                if center_y < h_limit_upper:
                    y_pos = "upper"
                elif center_y > h_limit_lower:
                    y_pos = "lower"
                else:
                    y_pos = "middle"
                    
                # Combine
                if x_pos == "center" and y_pos == "middle":
                    return "directly in front of you"
                elif x_pos == "center" and y_pos == "upper":
                    return "directly above you"
                elif x_pos == "center" and y_pos == "lower":
                    return "directly below you"
                elif y_pos == "upper":
                    return f"in the upper {x_pos}"
                elif y_pos == "lower":
                    return f"in the lower {x_pos}"
                else:
                    return f"to your {x_pos}"

            for d in detections:
                # Avoid repeating hazards in spatial layout if we already detailed them
                is_hazard = any(h.label == d.label for h in hazards if h.severity in ["medium", "high"])
                if is_hazard:
                    continue

                pos = get_spatial_position(d.box)
                if pos not in position_groups:
                    position_groups[pos] = []
                position_groups[pos].append(d)

            # 3. Format descriptions for each region
            parts = []
            if warnings_str:
                parts.append(f"Caution: {warnings_str}")

            # Add caption context
            clean_caption = filtered_caption.strip()
            if clean_caption:
                # Capitalize first letter and strip trailing dot
                if clean_caption.endswith("."):
                    clean_caption = clean_caption[:-1]
                parts.append(f"The scene shows {clean_caption[0].lower() + clean_caption[1:]}.")

            # Describe object layout
            layout_parts = []
            
            def format_group_with_prefix(prefix: str, items: List[DetectionItem]) -> str:
                if not items:
                    return ""
                
                high_conf = [d.label for d in items if d.confidence >= 0.75]
                low_conf = [d.label for d in items if d.confidence < 0.75]
                
                def format_labels(labels: List[str]) -> tuple:
                    if not labels:
                        return "", False
                    counts = {}
                    for label in labels:
                        counts[label] = counts.get(label, 0) + 1
                    
                    phrases = []
                    is_plural = False
                    total_count = 0
                    for label, count in counts.items():
                        total_count += count
                        if count > 1:
                            plural = label + "s" if not label.endswith("s") else label
                            phrases.append(f"{count} {plural}")
                            is_plural = True
                        else:
                            prefix_a = "an" if label[0] in "aeiou" else "a"
                            phrases.append(f"{prefix_a} {label}")
                    
                    if total_count > 1:
                        is_plural = True
                        
                    if len(phrases) == 1:
                        return phrases[0], is_plural
                    elif len(phrases) == 2:
                        return f"{phrases[0]} and {phrases[1]}", is_plural
                    else:
                        return ", ".join(phrases[:-1]) + f", and {phrases[-1]}", is_plural
                
                high_str, high_plural = format_labels(high_conf)
                low_str, _ = format_labels(low_conf)
                
                if high_str and low_str:
                    verb = "there are" if high_plural else "there is"
                    return f"{prefix}, {verb} {high_str}; and I think there may be {low_str}"
                elif high_str:
                    verb = "there are" if high_plural else "there is"
                    return f"{prefix}, {verb} {high_str}"
                else:
                    return f"{prefix}, I think there may be {low_str}"

            # Logical order for narration sequence
            order = [
                "directly in front of you",
                "to your left",
                "to your right",
                "directly above you",
                "directly below you",
                "in the upper left",
                "in the lower left",
                "in the upper right",
                "in the lower right"
            ]
            for pos in order:
                if pos in position_groups:
                    layout_parts.append(format_group_with_prefix(pos, position_groups[pos]))

            if layout_parts:
                parts.append("Specifically, " + "; and ".join(layout_parts) + ".")
            elif not warnings_str:
                parts.append("I cannot confidently identify any objects in the scene.")

            return " ".join(parts)
