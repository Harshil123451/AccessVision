from typing import List
from app.services.base import BaseService
from app.schemas.detect import DetectionItem
from app.schemas.scene import HazardItem
import logging

logger = logging.getLogger("accessvision")

class NarrationService(BaseService):
    """Combines detections, scene captioning, and hazards to generate descriptive and hazard-aware narration."""

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
            # 1. Handle critical hazards first to warn the user immediately
            hazard_warnings = []
            for h in hazards:
                if h.severity in ["medium", "high"]:
                    hazard_warnings.append(h.description)
                    
            warnings_str = " ".join(hazard_warnings)

            # 2. Group non-hazardous or general objects by location
            left_objects = []
            center_objects = []
            right_objects = []

            for d in detections:
                # Avoid repeating hazards in spatial layout if we already detailed them
                is_hazard = any(h.label == d.label for h in hazards if h.severity in ["medium", "high"])
                if is_hazard:
                    continue

                # Determine position based on bounding box center x
                # Box is [xmin, ymin, xmax, ymax]
                xmin, _, xmax, _ = d.box
                center_x = (xmin + xmax) / 2.0
                
                # Normalization scale: YOLO coords can be normalized or pixels.
                is_normalized = all(0.0 <= c <= 1.0 for c in d.box)
                boundary_left = 0.33 if is_normalized else 640 * 0.33
                boundary_right = 0.66 if is_normalized else 640 * 0.66

                if center_x < boundary_left:
                    left_objects.append(d.label)
                elif center_x > boundary_right:
                    right_objects.append(d.label)
                else:
                    center_objects.append(d.label)

            # 3. Format descriptions for each region
            parts = []
            if warnings_str:
                parts.append(f"Caution: {warnings_str}")

            # Add caption context
            clean_caption = caption.strip()
            if clean_caption:
                # Capitalize first letter and strip trailing dot
                if clean_caption.endswith("."):
                    clean_caption = clean_caption[:-1]
                parts.append(f"The scene shows {clean_caption[0].lower() + clean_caption[1:]}.")

            # Describe object layout
            layout_parts = []
            
            def format_group(items: List[str]) -> str:
                if not items:
                    return ""
                # Count occurrences to say "two chairs" instead of "chair and chair"
                counts = {}
                for item in items:
                    counts[item] = counts.get(item, 0) + 1
                
                phrases = []
                for item, count in counts.items():
                    if count > 1:
                        # Very basic pluralization helper
                        plural = item + "s" if not item.endswith("s") else item
                        phrases.append(f"{count} {plural}")
                    else:
                        # handle a/an
                        prefix = "an" if item[0] in "aeiou" else "a"
                        phrases.append(f"{prefix} {item}")
                
                if len(phrases) == 1:
                    return phrases[0]
                elif len(phrases) == 2:
                    return f"{phrases[0]} and {phrases[1]}"
                else:
                    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

            if center_objects:
                layout_parts.append(f"directly in front of you, there is {format_group(center_objects)}")
            if left_objects:
                layout_parts.append(f"to your left, there is {format_group(left_objects)}")
            if right_objects:
                layout_parts.append(f"to your right, there is {format_group(right_objects)}")

            if layout_parts:
                parts.append("Specifically, " + "; and ".join(layout_parts) + ".")
            elif not warnings_str:
                parts.append("No clear objects are detected in the immediate layout.")

            return " ".join(parts)
