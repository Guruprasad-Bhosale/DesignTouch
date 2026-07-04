import os
import sys
import numpy as np

# Adjust path so test runner finds app folder correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.config_service import ConfigService
from app.services.storage_service import StorageService
from app.services.tracking_service import TrackingService
from app.modules.sign_language.feature_extractor import FeatureExtractor
from app.modules.sign_language.classifier import SignLanguageClassifier
from app.modules.sign_language.language_processor import LanguageProcessor

def test_config_service(tmp_path):
    # Tests default configuration generation and loading
    config_file = tmp_path / "test_config.json"
    config = ConfigService(str(config_file))
    
    assert config.get("camera_index") == 0
    assert config.get("fps_cap") == 60
    assert "ThermalVision" in config.get("enabled_filters")
    
    # Verify save overrides
    config.set("camera_index", 2)
    assert config.get("camera_index") == 2
    
    # Reload and verify persistence
    config2 = ConfigService(str(config_file))
    assert config2.get("camera_index") == 2

def test_storage_service(tmp_path):
    # Tests SQL database schema, settings, and landmark dataset storage
    db_file = tmp_path / "test_storage.db"
    storage = StorageService(str(db_file))
    
    # Test setting storage
    storage.set_setting("cursor_speed", 2.5)
    assert storage.get_setting("cursor_speed") == 2.5
    
    # Test saving mock landmarks
    mock_lms = [np.array([float(i), float(i*2), float(i*3)]) for i in range(21)]
    storage.save_landmarks("A", mock_lms, "v1.0")
    
    # Verify retrieval
    dataset = storage.get_landmarks_dataset("v1.0")
    assert len(dataset) == 1
    assert dataset[0][0] == "A"
    # Coordinate values checks
    assert dataset[0][1] == 0.0 # x0
    assert dataset[0][2] == 0.0 # y0
    assert dataset[0][4] == 1.0 # x1
    
    # Verify distinct versions
    versions = storage.get_dataset_versions()
    assert "v1.0" in versions

def test_tracking_smoothing_and_debounce():
    # Tests virtual cursor smoothing and left hand pinch debounce click
    tracker = TrackingService("hand_landmarker.task")
    tracker.sensitivity = 1.0
    tracker.selection_threshold = 0.035
    tracker.release_threshold = 0.055
    
    # Set mock right hand index tip (8) coordinates
    tracker._hand_present["Right"] = True
    tracker._smoothed_landmarks["Right"] = {
        8: np.array([0.4, 0.3, 0.0])
    }
    
    # Run gesture checks simulating frames
    w, h = 1280, 720
    # Simulate coordinate moving average buffer filling up (8 steps)
    for _ in range(10):
        tracker._detect_gestures(w, h)
        
    cursor_x, cursor_y = tracker.cursor_position
    # Right index tip raw coordinate x = 0.4.
    # Scaled to screen: 1280 * 0.4 = 512. Y coordinate = 720 * 0.3 = 216.
    assert cursor_x == 512
    assert cursor_y == 216
    
    # Test Left hand pinch click and release thresholds
    tracker._hand_present["Left"] = True
    # Thumb (4) and Index (8)
    # 1. Open pinch: distance = 0.1
    tracker._smoothed_landmarks["Left"] = {
        4: np.array([0.1, 0.1, 0.0]),
        8: np.array([0.2, 0.1, 0.0])
    }
    tracker._detect_gestures(w, h)
    assert tracker.click_state == False
    assert tracker._is_pinched == False
    
    # 2. Closed pinch below 0.035: distance = 0.02
    tracker._smoothed_landmarks["Left"] = {
        4: np.array([0.1, 0.1, 0.0]),
        8: np.array([0.12, 0.1, 0.0])
    }
    tracker._detect_gestures(w, h)
    assert tracker.click_state == True  # Rising edge click event triggers
    assert tracker._is_pinched == True
    
    # 3. Successive closed pinch: click must debounce (return False)
    tracker._detect_gestures(w, h)
    assert tracker.click_state == False  # Debounced
    assert tracker._is_pinched == True
    
    # 4. Partial release below release threshold 0.055: distance = 0.045
    tracker._smoothed_landmarks["Left"] = {
        4: np.array([0.1, 0.1, 0.0]),
        8: np.array([0.145, 0.1, 0.0])
    }
    tracker._detect_gestures(w, h)
    assert tracker.click_state == False
    assert tracker._is_pinched == True  # Remains pinched (hysteresis)
    
    # 5. Full release above release threshold 0.055: distance = 0.07
    tracker._smoothed_landmarks["Left"] = {
        4: np.array([0.1, 0.1, 0.0]),
        8: np.array([0.17, 0.1, 0.0])
    }
    tracker._detect_gestures(w, h)
    assert tracker.click_state == False
    assert tracker._is_pinched == False  # Released!

def test_feature_extractor():
    extractor = FeatureExtractor()
    
    # Create mock hand landmarks
    landmarks = {}
    # Wrist (0) at origin
    landmarks[0] = np.array([0.0, 0.0, 0.0])
    # Set coordinates for all other points
    for idx in range(1, 21):
        landmarks[idx] = np.array([float(idx)*0.01, float(idx)*0.02, 0.1])
        
    features = extractor.extract_features(landmarks)
    # Output must have exactly 103 dimensions
    assert features.shape == (103,)
    assert isinstance(features, np.ndarray)

def test_language_processor():
    processor = LanguageProcessor(debounce_frames=2, stable_time_threshold=0.1)
    
    # 1. Process noisy character (should not trigger append)
    has_new, text, sugs = processor.process_prediction("A", 0.5) # low confidence
    assert has_new == False
    assert processor.current_text == ""
    
    import time
    def feed_stable_char(char, conf=0.85):
        # Feed 6 frames of the character to establish majority vote and set last_char
        for _ in range(6):
            processor.process_prediction(char, conf)
        # Sleep to exceed the stable time threshold
        time.sleep(0.15)
        # Final frame to trigger and confirm
        return processor.process_prediction(char, conf)
    
    # 2. Process stable character
    has_new, text, sugs = feed_stable_char("H")
    assert has_new == True
    assert text == "H"
    
    # Add other letters to build word
    feed_stable_char("E")
    feed_stable_char("L")
    
    # Reset temporal hold state to allow second L by filling with "None"
    for _ in range(6):
        processor.process_prediction("None", 0.5)
        
    feed_stable_char("L")
    feed_stable_char("O")
    
    assert processor.current_word == "HELLO"
    
    # Test autocorrect prediction suggestions matching prefix
    sugs = processor.get_suggestions(processor.current_word)
    assert "HELLO" in sugs
    
    # Test space commits word
    processor.add_character("SPACE")
    assert processor.current_text == "HELLO "
    assert processor.current_word == ""

def test_classifier_heuristics():
    classifier = SignLanguageClassifier(model_path="nonexistent_model.pkl")
    
    # 1. Closed fist - 'A'
    fv_a = np.zeros(103, dtype=np.float32)
    # t_dist, i_dist, m_dist, r_dist, p_dist
    fv_a[63:68] = [1.1, 1.1, 1.1, 1.1, 1.1]
    fv_a[68] = 0.5
    char, conf = classifier.predict(fv_a)
    assert char == "A"
    assert conf == 0.80
    
    # 2. Open palm - 'B'
    fv_b = np.zeros(103, dtype=np.float32)
    fv_b[63:68] = [1.6, 1.8, 1.8, 1.8, 1.8]
    fv_b[68] = 1.1
    char, conf = classifier.predict(fv_b)
    assert char == "B"
    assert conf == 0.85
    
    # 3. Peace sign - 'V'
    fv_v = np.zeros(103, dtype=np.float32)
    fv_v[63:68] = [1.2, 1.8, 1.8, 1.1, 1.1]
    char, conf = classifier.predict(fv_v)
    assert char == "V"
    assert conf == 0.80
    
    # 4. Pinky and Thumb extended - 'Y'
    fv_y = np.zeros(103, dtype=np.float32)
    fv_y[63:68] = [1.6, 1.1, 1.1, 1.1, 1.8]
    char, conf = classifier.predict(fv_y)
    assert char == "Y"
    assert conf == 0.85
    
    # 5. Cup hand - 'C'
    fv_c = np.zeros(103, dtype=np.float32)
    fv_c[63:68] = [1.3, 1.4, 1.4, 1.4, 1.4]
    fv_c[68] = 0.8
    char, conf = classifier.predict(fv_c)
    assert char == "C"
    assert conf == 0.75

