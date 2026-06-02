import asyncio
from collections import deque
from typing import List, Dict, Any, Optional
from app.services.base import BaseService
from app.services.detect_service import DetectService
from app.services.caption_service import CaptionService
from app.services.narration_service import NarrationService
from app.schemas.scene import SceneAnalysisResult, HazardItem
from app.schemas.detect import DetectionItem
import logging

logger = logging.getLogger("accessvision")

class SceneService(BaseService):
    """Orchestrates overall scene understanding, combining object detection,
    captioning, hazard analysis, and accessibility narration. Maintains a rolling memory
    of recent scene states.
    """

    # Static rolling memory queue (stores up to last 10 analyzed scenes)
    _memory = deque(maxlen=10)

    def __init__(self):
        self.detect_service = DetectService()
        self.caption_service = CaptionService()
        self.narration_service = NarrationService()

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
            # We check if coordinate ymax is in the bottom region of the image (e.g. ymax > 400 pixels or > 0.65 normalized)
            # which indicates close proximity to the ground/feet.
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

    async def analyze_scene(self, image_bytes: bytes, detections: Optional[List[DetectionItem]] = None, is_mirrored: bool = False) -> SceneAnalysisResult:
        """Runs image captioning and object detection. If pre-computed detections are provided,
        reuses them to bypass redundant object detection inference.
        """
        logger.info("Starting unified scene analysis")
        
        with self.measure_latency() as metrics:
            caption_task = self.caption_service.generate_caption(image_bytes)
            
            if detections is None:
                # Run captioning and detection concurrently to reduce latency
                detect_task = self.detect_service.detect_objects(image_bytes)
                caption_res, detect_res = await asyncio.gather(caption_task, detect_task)
                caption = caption_res.caption if caption_res.success else ""
                detections = detect_res.detections if detect_res.success else []
            else:
                # Re-use pre-computed detections and run only captioning
                logger.info("[CACHE] Reused detections in scene analysis")
                caption_res = await caption_task
                caption = caption_res.caption if caption_res.success else ""

            # Assess hazards
            hazards = self._assess_hazards(detections)

            if is_mirrored:
                logger.info("[CAMERA] Mirrored preview enabled")
            else:
                logger.info("[CAMERA] Mirrored preview disabled")

            # Generate narration using grounded perception details
            from app.utils.image import load_image_from_bytes
            pil_image = load_image_from_bytes(image_bytes)
            try:
                narration = self.narration_service.generate_narration(
                    caption=caption,
                    detections=detections,
                    hazards=hazards,
                    pil_image=pil_image,
                    recent_memory=self.get_recent_memory()
                )
            finally:
                pil_image.close()

        result = SceneAnalysisResult(
            success=True,
            caption=caption,
            objects=detections,
            hazards=hazards,
            narration=narration,
            metrics={"inference_ms": metrics["latency_ms"]}
        )

        # Store in rolling scene memory
        self._memory.append({
            "timestamp": asyncio.get_event_loop().time(),
            "caption": caption,
            "detections": [d.model_dump() for d in detections],
            "hazards": [h.model_dump() for h in hazards],
            "narration": narration
        })

        return result

    @classmethod
    def get_recent_memory(cls) -> List[Dict[str, Any]]:
        """Returns the rolling memory queue contents."""
        return list(cls._memory)
