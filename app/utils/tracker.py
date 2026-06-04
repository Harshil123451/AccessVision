import time
from typing import List, Dict, Any, Optional

class TrackedObject:
    """Represents an object track across sequential frames."""
    def __init__(self, label: str, box: List[float], confidence: float, color: str, timestamp: float):
        self.label = label
        self.box = box
        self.confidence = confidence
        self.color = color
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.frames_active = 1
        self.frames_inactive = 0
        self.is_decaying = False

class TemporalConsistencyTracker:
    """Smooths out detection frame-to-frame fluctuations using IoU matching and confidence decay."""
    def __init__(self, max_inactive_frames: int = 3, iou_threshold: float = 0.3, decay_rate: float = 0.75):
        self.tracked_objects: List[TrackedObject] = []
        self.max_inactive_frames = max_inactive_frames
        self.iou_threshold = iou_threshold
        self.decay_rate = decay_rate

    def update(self, current_detections: List[Dict[str, Any]], current_colors: List[str], current_time: float) -> List[Dict[str, Any]]:
        """Updates tracked objects and returns currently stable/active detections."""
        
        def calculate_iou(b1: List[float], b2: List[float]) -> float:
            # Box format: [xmin, ymin, xmax, ymax]
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

        matched_indices = set()
        updated_tracks = []
        
        # 1. Match current detections with existing tracks
        for track in self.tracked_objects:
            best_iou = 0.0
            best_idx = -1
            
            for idx, det in enumerate(current_detections):
                if idx in matched_indices:
                    continue
                if det["label"].lower() != track.label.lower():
                    continue
                
                iou = calculate_iou(track.box, det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= self.iou_threshold and best_idx != -1:
                det = current_detections[best_idx]
                # Found match: update state
                track.box = det["box"]
                # Rolling confidence smoothing
                track.confidence = 0.7 * track.confidence + 0.3 * det["confidence"]
                track.color = current_colors[best_idx] if best_idx < len(current_colors) else det.get("color", "unknown")
                track.last_seen = current_time
                track.frames_active += 1
                track.frames_inactive = 0
                track.is_decaying = False
                matched_indices.add(best_idx)
                updated_tracks.append(track)
            else:
                # No match: increment inactivity counter
                track.frames_inactive += 1
                if track.frames_inactive <= self.max_inactive_frames:
                    # Decay confidence exponentially to prevent sudden disappearance
                    track.confidence *= self.decay_rate
                    track.is_decaying = True
                    updated_tracks.append(track)
                    
        # 2. Register completely new detections
        for idx, det in enumerate(current_detections):
            if idx in matched_indices:
                continue
            color = current_colors[idx] if idx < len(current_colors) else det.get("color", "unknown")
            new_track = TrackedObject(
                label=det["label"],
                box=det["box"],
                confidence=det["confidence"],
                color=color,
                timestamp=current_time
            )
            updated_tracks.append(new_track)
            
        self.tracked_objects = updated_tracks
        
        # 3. Filter active tracks to return to caller
        results = []
        for track in self.tracked_objects:
            # Suppress very low confidence tracks
            if track.confidence >= 0.20:
                results.append({
                    "label": track.label,
                    "box": track.box,
                    "confidence": round(track.confidence, 4),
                    "color": track.color,
                    "frames_active": track.frames_active,
                    "is_decaying": track.is_decaying
                })
        return results
