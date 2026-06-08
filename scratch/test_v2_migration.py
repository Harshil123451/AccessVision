import asyncio
import sys
import os
import time

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.tracker import TemporalConsistencyTracker
from app.services.fast_narration_service import FastNarrationService
from app.services.enrichment_cache import enrichment_cache
from app.services.question_router_service import QuestionRouterService
from app.schemas.perception import PerceptionGraph, GroundedObject
from app.schemas.scene import HazardItem

def test_tracker_and_fast_narration():
    print("\n--- Testing Tracker and Fast Narration Engine ---")
    tracker = TemporalConsistencyTracker(max_inactive_frames=3, iou_threshold=0.3)
    fast_narration = FastNarrationService()
    
    # Frame 1: Suitcase appears
    t1 = time.time()
    dets_f1 = [{"label": "suitcase", "box": [0.1, 0.7, 0.3, 0.9], "confidence": 0.85}] # bottom-left (on the floor)
    current_colors_f1 = ["blue"]
    
    res_f1 = tracker.update(dets_f1, current_colors_f1, t1)
    print(f"Frame 1 - Active tracks: {len(res_f1)}")
    print(f"Frame 1 - Appearances: {[a['label'] for a in tracker.appearances]}")
    print(f"Frame 1 - Disappearances: {tracker.disappearances}")
    print(f"Frame 1 - Movements: {tracker.movements}")
    print(f"Frame 1 - Scene stable: {tracker.scene_stable}")
    
    assert len(tracker.appearances) == 1
    assert tracker.appearances[0]["label"] == "suitcase"
    assert not tracker.scene_stable
    
    pg_f1 = fast_narration.build_perception_graph(res_f1, 640, 640)
    narration_f1 = fast_narration.generate_narration(pg_f1, tracker, [], is_first_frame=True)
    print(f"Frame 1 - Full Narration: \"{narration_f1}\"")
    assert "blue suitcase" in narration_f1.lower()
    assert "on the floor to your left" in narration_f1.lower()
    
    # Frame 2: Suitcase stays, chair appears
    t2 = t1 + 0.5
    dets_f2 = [
        {"label": "suitcase", "box": [0.1, 0.7, 0.3, 0.9], "confidence": 0.85},
        {"label": "chair", "box": [0.4, 0.4, 0.6, 0.6], "confidence": 0.90} # middle-center
    ]
    current_colors_f2 = ["blue", "brown"]
    
    res_f2 = tracker.update(dets_f2, current_colors_f2, t2)
    print(f"\nFrame 2 - Active tracks: {len(res_f2)}")
    print(f"Frame 2 - Appearances: {[a['label'] for a in tracker.appearances]}")
    print(f"Frame 2 - Disappearances: {tracker.disappearances}")
    print(f"Frame 2 - Movements: {tracker.movements}")
    print(f"Frame 2 - Scene stable: {tracker.scene_stable}")
    
    assert len(tracker.appearances) == 1
    assert tracker.appearances[0]["label"] == "chair"
    
    pg_f2 = fast_narration.build_perception_graph(res_f2, 640, 640)
    narration_f2 = fast_narration.generate_narration(pg_f2, tracker, [], is_first_frame=False)
    print(f"Frame 2 - Change Narration: \"{narration_f2}\"")
    assert "chair" in narration_f2.lower()
    assert "suitcase" not in narration_f2.lower() # Unchanged suitcase should be omitted!
    
    # Frame 3: Stable frame
    t3 = t2 + 0.5
    res_f3 = tracker.update(dets_f2, current_colors_f2, t3)
    print(f"\nFrame 3 - Active tracks: {len(res_f3)}")
    print(f"Frame 3 - Appearances: {[a['label'] for a in tracker.appearances]}")
    print(f"Frame 3 - Disappearances: {tracker.disappearances}")
    print(f"Frame 3 - Movements: {tracker.movements}")
    print(f"Frame 3 - Scene stable: {tracker.scene_stable}")
    
    assert tracker.scene_stable
    pg_f3 = fast_narration.build_perception_graph(res_f3, 640, 640)
    narration_f3 = fast_narration.generate_narration(pg_f3, tracker, [], is_first_frame=False)
    print(f"Frame 3 - Stable Narration: \"{narration_f3}\"")
    assert narration_f3 == "Scene unchanged."
    
    # Frame 4: Chair moves to left
    t4 = t3 + 0.5
    dets_f4 = [
        {"label": "suitcase", "box": [0.1, 0.7, 0.3, 0.9], "confidence": 0.85},
        {"label": "chair", "box": [0.32, 0.4, 0.52, 0.6], "confidence": 0.90} # center shifted from 0.5 to 0.42 (to the left)
    ]
    current_colors_f4 = ["blue", "brown"]
    
    res_f4 = tracker.update(dets_f4, current_colors_f4, t4)
    print(f"\nFrame 4 - Active tracks: {len(res_f4)}")
    print(f"Frame 4 - Appearances: {[a['label'] for a in tracker.appearances]}")
    print(f"Frame 4 - Disappearances: {tracker.disappearances}")
    print(f"Frame 4 - Movements: {tracker.movements}")
    print(f"Frame 4 - Scene stable: {tracker.scene_stable}")
    
    assert len(tracker.movements) == 1
    assert "moved to your left" in tracker.movements[0]
    
    pg_f4 = fast_narration.build_perception_graph(res_f4, 640, 640)
    narration_f4 = fast_narration.generate_narration(pg_f4, tracker, [], is_first_frame=False)
    print(f"Frame 4 - Move Narration: \"{narration_f4}\"")
    assert "moved to your left" in narration_f4.lower()
    
    # Frame 5: Chair disappears
    t5 = t4 + 0.5
    dets_f5 = [
        {"label": "suitcase", "box": [0.1, 0.7, 0.3, 0.9], "confidence": 0.85}
    ]
    current_colors_f5 = ["blue"]
    
    # Run three updates to let confidence decay or check immediate decay event
    res_f5 = tracker.update(dets_f5, current_colors_f5, t5)
    print(f"\nFrame 5 - Active tracks: {len(res_f5)}")
    print(f"Frame 5 - Appearances: {tracker.appearances}")
    print(f"Frame 5 - Disappearances: {tracker.disappearances}")
    
    assert "chair" in tracker.disappearances
    pg_f5 = fast_narration.build_perception_graph(res_f5, 640, 640)
    narration_f5 = fast_narration.generate_narration(pg_f5, tracker, [], is_first_frame=False)
    print(f"Frame 5 - Disappearance Narration: \"{narration_f5}\"")
    assert "no longer visible" in narration_f5.lower()
    print("Tracker and Fast Narration tests passed!")

def test_enrichment_cache():
    print("\n--- Testing Background Enrichment Cache ---")
    sid = "test_session_123"
    
    # Clean cache
    enrichment_cache.set(sid, "florence_caption", "A clean room with a bed.")
    enrichment_cache.set(sid, "ocr_result", "Stop Sign")
    
    cap = enrichment_cache.get(sid, "florence_caption")
    ocr = enrichment_cache.get(sid, "ocr_result")
    
    assert cap == "A clean room with a bed."
    assert ocr == "Stop Sign"
    assert enrichment_cache.hits == 2
    
    # Test session isolation
    cap_other = enrichment_cache.get("other_session", "florence_caption")
    assert cap_other is None
    assert enrichment_cache.misses == 1
    
    # Test TTL expiration simulation
    enrichment_cache.set(sid, "short_lived", "Expired soon", ttl=-1) # Expired immediately
    expired = enrichment_cache.get(sid, "short_lived")
    assert expired is None
    
    print("Background Enrichment Cache tests passed!")

def test_question_routing():
    print("\n--- Testing Question Router Intents ---")
    router = QuestionRouterService()
    
    tests = [
        ("What is in front of me?", "FAST_OBJECT_QUERY"),
        ("What objects do you see?", "FAST_OBJECT_QUERY"),
        ("Is there a chair?", "FAST_OBJECT_QUERY"),
        ("What color is the suitcase?", "FAST_COLOR_QUERY"),
        ("What color is the bed?", "FAST_COLOR_QUERY"),
        ("Read this label", "OCR_QUERY"),
        ("What does this sign say?", "OCR_QUERY"),
        ("Describe the room", "DETAILED_SCENE_QUERY"),
        ("What do you see?", "DETAILED_SCENE_QUERY"),
        ("Is the bed made?", "VQA_QUERY"),
        ("What is on the table?", "VQA_QUERY"),
    ]
    
    for question, expected_intent in tests:
        intent, target = router.analyze_question(question)
        print(f"Question: '{question}' -> Classified: {intent} (Expected: {expected_intent})")
        assert intent == expected_intent, f"Failed for '{question}': got {intent}, expected {expected_intent}"
        
    print("Question Router tests passed!")

if __name__ == "__main__":
    print("=========================================")
    print("Running AccessVision V2 Migration Tests")
    print("=========================================")
    test_tracker_and_fast_narration()
    test_enrichment_cache()
    test_question_routing()
    print("\nAll V2 validation checks passed successfully!")

