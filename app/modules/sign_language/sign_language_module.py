import numpy as np
import time
from app.modules.base_module import BaseModule
from app.core.interfaces import ITrackingService, IStorageService, IConfigService
from app.modules.sign_language.feature_extractor import FeatureExtractor
from app.modules.sign_language.classifier import SignLanguageClassifier
from app.modules.sign_language.data_collector import SignLanguageDataCollector
from app.modules.sign_language.language_processor import LanguageProcessor
from app.modules.sign_language.text_output import SignLanguageTextOutput

class SignLanguageModule(BaseModule):
    def __init__(self):
        self._service_manager = None
        self._tracker = None
        self._storage = None
        self._config = None
        
        # Pipeline stages
        self.extractor = None
        self.classifier = None
        self.collector = None
        self.processor = None
        self.renderer = None
        
        # Live state variables
        self.active_char = "None"
        self.confidence = 0.0
        self.suggestions = []
        self.current_sentence = ""
        
        # Dataset collection parameters
        self.collection_active = False
        self.target_label = "A"
        self.dataset_version = "v1.0"
        self._last_record_time = 0.0
        self._record_cooldown = 0.35  # Record one sample every 350 ms
        self.recorded_count = 0
        
        # Prediction rate-limiting variables
        self._last_predict_time = 0.0
        self._predict_interval = 0.05  # 50 ms (20 Hz)

    def initialize(self, service_manager) -> bool:
        self._service_manager = service_manager
        try:
            self._tracker = service_manager.get(ITrackingService)
            self._storage = service_manager.get(IStorageService)
            self._config = service_manager.get(IConfigService)
            
            # 1. Initialize Pipeline Stages
            self.extractor = FeatureExtractor()
            
            model_path = self._config.get("sign_language_model", "app/models/sign_language_cls.pkl")
            self.classifier = SignLanguageClassifier(model_path)
            
            self.collector = SignLanguageDataCollector(self._storage)
            self.processor = LanguageProcessor()
            self.renderer = SignLanguageTextOutput()
            
            self.dataset_version = self._config.get("dataset_version", "v1.0")
            
            print("[SignLanguageModule] Initialized successfully.")
            return True
        except Exception as e:
            print(f"[SignLanguageModule Error] Initialization failed: {e}")
            return False

    def process_frame(self, frame, tracking_data: dict) -> np.ndarray:
        # 1. Get raw landmarks (Right hand index/palm is used for alphabet prediction)
        landmarks = self._tracker.smoothed_landmarks
        presence = self._tracker.hand_present
        
        has_hand = presence["Right"] and len(landmarks["Right"]) >= 21
        
        # In case only left hand is shown, check if we want to fallback to it for prediction
        if not has_hand and presence["Left"] and len(landmarks["Left"]) >= 21:
            hand_lms = landmarks["Left"]
            has_hand = True
        else:
            hand_lms = landmarks["Right"] if has_hand else None
            
        if has_hand and hand_lms:
            # 2. Extract relative and scaled features
            features = self.extractor.extract_features(hand_lms)
            
            # 3. Classify features (rate-limited to 20Hz)
            t_now = time.time()
            if t_now - self._last_predict_time >= self._predict_interval:
                self.active_char, self.confidence = self.classifier.predict(features)
                self._last_predict_time = t_now
            
            # 4. Perform live dataset collection if active
            if self.collection_active:
                if t_now - self._last_record_time >= self._record_cooldown:
                    success = self.collector.record_sample(self.target_label, hand_lms, self.dataset_version)
                    if success:
                        self.recorded_count += 1
                        self._last_record_time = t_now
                        print(f"[SignLanguageModule] Recorded sample {self.recorded_count} for character '{self.target_label}'")
            
            # 5. Process continuous sentences
            has_new, self.current_sentence, self.suggestions = self.processor.process_prediction(
                self.active_char, self.confidence
            )
        else:
            self.active_char = "None"
            self.confidence = 0.0
            
        # 6. Render UI text overlays on top of the frame
        frame = self.renderer.draw_predictions(
            frame, 
            self.processor.current_text, 
            self.processor.current_word, 
            self.active_char, 
            self.confidence, 
            self.suggestions
        )
        
        return frame

    def handle_gesture(self, gesture_event: str):
        if gesture_event == "clear_text":
            self.processor.clear()
            self.current_sentence = ""
            print("[SignLanguageModule] Sentence cleared.")
        elif gesture_event == "commit_word":
            self.processor.commit_word()
            print("[SignLanguageModule] Word committed.")

    def trigger_collection(self, label: str, duration_sec=5.0):
        """Toggles active capture recording in the process loop."""
        self.target_label = label.upper()
        self.recorded_count = 0
        self.collection_active = True
        self._last_record_time = time.time()
        print(f"[SignLanguageModule] Collection started for label: {self.target_label}")

    def stop_collection(self):
        self.collection_active = False
        print(f"[SignLanguageModule] Collection stopped. Total recorded: {self.recorded_count}")

    def train_model(self) -> bool:
        """Loads landmarks from SQLite database and retrains the model."""
        try:
            records = self._storage.get_landmarks_dataset(self.dataset_version)
            if not records:
                print(f"[SignLanguageModule Error] No samples found in database for version {self.dataset_version}")
                return False
                
            features_list = []
            labels = []
            
            for row in records:
                label = row[0]
                coords = list(row[1:])
                # Reconstruct landmarks mapping for FeatureExtractor
                lms_dict = {}
                for idx in range(21):
                    x = coords[idx * 3]
                    y = coords[idx * 3 + 1]
                    z = coords[idx * 3 + 2]
                    lms_dict[idx] = np.array([x, y, z])
                    
                features = self.extractor.extract_features(lms_dict)
                features_list.append(features)
                labels.append(label)
                
            X = np.array(features_list)
            success = self.classifier.train(X, labels)
            
            # Reload classifier model
            if success:
                self.classifier.load()
            return success
        except Exception as e:
            print(f"[SignLanguageModule Error] Retraining failed: {e}")
            return False

    def get_ui_widget(self):
        return None

    def deinitialize(self):
        self.collection_active = False
        print("[SignLanguageModule] Deinitialized.")
