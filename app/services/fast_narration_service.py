import logging
from typing import List, Dict, Any, Optional
from app.schemas.perception import PerceptionGraph, GroundedObject
from app.schemas.detect import DetectionItem
from app.schemas.scene import HazardItem
from app.utils.tracker import TemporalConsistencyTracker

logger = logging.getLogger("accessvision")

class FastNarrationService:
    """Fast Narration Engine that generates immediate verbal narration (<500ms)
    relying only on the YOLO Perception Graph and Temporal Tracker state.
    """
    
    @staticmethod
    def get_spatial_description(box: List[float], label: str, w: int, h: int) -> str:
        """Translates coordinates into natural spatial positions and approximate distances."""
        xmin, ymin, xmax, ymax = box
        
        # Handle normalized vs pixel coordinates
        is_normalized = all(0.0 <= c <= 1.0 for c in box)
        if is_normalized:
            norm_xmin, norm_ymin, norm_xmax, norm_ymax = xmin, ymin, xmax, ymax
        else:
            norm_xmin = xmin / w
            norm_ymin = ymin / h
            norm_xmax = xmax / w
            norm_ymax = ymax / h
            
        center_x = (norm_xmin + norm_xmax) / 2.0
        ymax_norm = norm_ymax
        area = (norm_xmax - norm_xmin) * (norm_ymax - norm_ymin)
        
        # 1. Horizontal position mapping
        if center_x < 0.33:
            x_pos = "to your left"
        elif center_x > 0.66:
            x_pos = "to your right"
        else:
            x_pos = "directly ahead"

        # 2. Distance mapping
        if area >= 0.20:
            distance_phrase = "less than a meter away"
        elif area >= 0.08:
            distance_phrase = "approximately one meter away"
        elif area >= 0.03:
            distance_phrase = "approximately two meters away"
        else:
            distance_phrase = "in the background"

        # 3. Floor position mapping for common floor objects
        floor_objects = {
            "suitcase", "backpack", "shoes", "chair", "toy", "dog", "cat", 
            "cup", "bottle", "bowl", "book", "handbag", "skateboard"
        }
        
        is_on_floor = ymax_norm > 0.65 and label.lower() in floor_objects
        
        if is_on_floor:
            if x_pos == "directly ahead":
                return "on the floor directly ahead"
            else:
                return f"on the floor {x_pos}"
        else:
            if x_pos == "directly ahead":
                if area >= 0.15:
                    return "directly ahead"
                else:
                    return f"directly ahead, {distance_phrase}"
            else:
                if distance_phrase == "in the background":
                    return f"in the background {x_pos}"
                else:
                    return f"{distance_phrase} {x_pos}"

    def build_perception_graph(
        self,
        tracked_results: List[Dict[str, Any]],
        w: int,
        h: int
    ) -> PerceptionGraph:
        """Constructs a PerceptionGraph from the temporal tracker's active tracks."""
        perception_objects = []
        for track in tracked_results:
            pos_desc = self.get_spatial_description(track["box"], track["label"], w, h)
            
            # Categorize size
            xmin, ymin, xmax, ymax = track["box"]
            
            # Calculate area based on normalized box coordinates
            is_normalized = all(0.0 <= c <= 1.0 for c in track["box"])
            if is_normalized:
                area = (xmax - xmin) * (ymax - ymin)
            else:
                area = ((xmax - xmin) * (ymax - ymin)) / (w * h)
                
            if area >= 0.20:
                size_desc = "large"
            elif area >= 0.04:
                size_desc = "medium"
            else:
                size_desc = "small"
                
            # Confidence tier
            conf = track["confidence"]
            if conf >= 0.75:
                tier = "HIGH"
            elif conf >= 0.50:
                tier = "MEDIUM"
            else:
                tier = "LOW"
                
            perception_objects.append(GroundedObject(
                class_name=track["label"],
                confidence=conf,
                color=track["color"],
                position=pos_desc,
                size=size_desc,
                grounding_source="YOLO",
                narration_confidence=tier
            ))
            
        return PerceptionGraph(objects=perception_objects)

    def generate_narration(
        self,
        perception_graph: PerceptionGraph,
        tracker: TemporalConsistencyTracker,
        hazards: List[HazardItem],
        is_first_frame: bool
    ) -> str:
        """Generates real-time verbal narration based on the current graph and tracker state."""
        parts = []

        # 1. Critical Hazards warnings
        hazard_warnings = [h.description for h in hazards if h.severity in ["medium", "high"]]
        if hazard_warnings:
            parts.append(f"Caution: {' '.join(hazard_warnings)}")

        # Helper to format object description
        def format_object(obj: GroundedObject, prefix: str = "") -> str:
            color_phrase = f"{obj.color} " if obj.color != "unknown" else ""
            label = obj.class_name
            a_an = "An" if label[0].lower() in 'aeiou' and not color_phrase else "A"
            if color_phrase:
                a_an = "An" if color_phrase[0].lower() in 'aeiou' else "A"
            
            # Adjust case for sentence starting vs continuation
            if prefix:
                a_an = a_an.lower()
                return f"{prefix} {a_an} {color_phrase}{label} is {obj.position}."
            else:
                return f"{a_an} {color_phrase}{label} is {obj.position}."

        # 2. Roll Out Scene Changes or Full Description
        if is_first_frame:
            # First frame: narrate all objects
            object_sentences = []
            for obj in perception_graph.objects:
                # Omit if it's already narrated as a high-severity hazard
                is_hazard = any(h.label.lower() == obj.class_name.lower() for h in hazards if h.severity in ["medium", "high"])
                if is_hazard:
                    continue
                object_sentences.append(format_object(obj))
            
            if object_sentences:
                parts.extend(object_sentences)
            elif not hazard_warnings:
                parts.append("The scene shows an environment. I cannot confidently identify any objects.")
        else:
            # Subsequent frame: describe changes only
            change_sentences = []

            # A. Disappearances
            for disp in tracker.disappearances:
                change_sentences.append(f"The {disp} is no longer visible.")

            # B. Movements
            for mov in tracker.movements:
                change_sentences.append(mov)

            # C. Appearances
            for app in tracker.appearances:
                # Find matching object in graph to get correct spatial/size formatting
                matching_obj = next((o for o in perception_graph.objects if o.class_name.lower() == app["label"].lower()), None)
                if matching_obj:
                    is_hazard = any(h.label.lower() == matching_obj.class_name.lower() for h in hazards if h.severity in ["medium", "high"])
                    if not is_hazard:
                        change_sentences.append(format_object(matching_obj))

            if change_sentences:
                parts.extend(change_sentences)
            elif tracker.scene_stable and not hazard_warnings:
                parts.append("Scene unchanged.")

        # If nothing generated at all, return default fallback
        if not parts:
            return "Scene unchanged."

        return " ".join(parts)

