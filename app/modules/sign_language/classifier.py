import os
import pickle
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from app.core.structured_logger import StructuredLogger

class SignLanguageClassifier:
    def __init__(self, model_path="app/models/sign_language_cls.pkl"):
        self.logger = StructuredLogger.get_logger("Classifier")
        self._model_path = model_path
        self._model = None
        self.label_encoder = None
        
        # Resolve absolute path
        self._resolve_model_path()
        self.load()

    def _resolve_model_path(self):
        if not os.path.isabs(self._model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self._model_path = os.path.join(base_dir, self._model_path)

    def load(self) -> bool:
        enable_ml = True
        try:
            from app.core.service_manager import ServiceManager
            from app.core.interfaces import IConfigService
            config_service = ServiceManager.get(IConfigService)
            enable_ml = config_service.get("enable_ml_classifier", True)
        except Exception:
            enable_ml = True

        if not enable_ml:
            self.logger.info("ML Classifier is disabled via config. Bypassing pickle load to use fallback heuristics.")
            self._model = None
            return False

        if os.path.exists(self._model_path):
            try:
                with open(self._model_path, "rb") as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and "model" in data:
                    self._model = data["model"]
                    self.label_encoder = data.get("label_encoder", None)
                else:
                    self._model = data
                    self.label_encoder = None
                    
                self.logger.info(f"Model loaded successfully from {self._model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load model file: {e}")
        else:
            self.logger.warning(f"Model file not found at {self._model_path}. Fallback prediction heuristics will be active.")
        return False

    def save(self):
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        try:
            with open(self._model_path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "label_encoder": self.label_encoder
                }, f)
            self.logger.info(f"Model saved successfully to {self._model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model file: {e}")
            return False

    def train(self, X: np.ndarray, y: list) -> bool:
        """Trains multiple candidate models, determines the best by validation score, and saves it."""
        if len(X) == 0 or len(y) == 0:
            self.logger.error("Training cancelled: empty dataset.")
            return False
            
        try:
            self.logger.info(f"Splitting dataset of {len(X)} samples into train and validation sets...")
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Stratify to ensure equal class proportions in train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "SVM": SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
                "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
            }
            
            # Evaluate XGBoost if installed
            try:
                from xgboost import XGBClassifier
                models["XGBoost"] = XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    eval_metric="mlogloss"
                )
            except ImportError:
                self.logger.warning("XGBoost is not installed in the environment. Skipping XGBoost evaluation.")
                
            best_name = None
            best_val_acc = -1.0
            best_model = None
            eval_results = {}
            
            for name, clf in models.items():
                self.logger.info(f"Fitting and evaluating {name}...")
                clf.fit(X_train, y_train)
                val_acc = clf.score(X_val, y_val)
                self.logger.info(f"{name} validation accuracy: {val_acc:.4f}")
                eval_results[name] = float(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_name = name
                    best_model = clf
                    
            self.logger.info(f"Best classifier selected: {best_name} (Val Acc: {best_val_acc:.4f})")
            
            # Re-fit the best model on the entire dataset for production-ready deployment
            self.logger.info(f"Re-fitting {best_name} on the entire dataset...")
            best_model.fit(X, y_encoded)
            
            self._model = best_model
            self.save()
            
            # Save evaluation results
            eval_dir = os.path.dirname(self._model_path)
            results_path = os.path.join(eval_dir, "evaluation_results.json")
            with open(results_path, "w") as f:
                json.dump({
                    "best_model": best_name,
                    "best_accuracy": float(best_val_acc),
                    "all_scores": eval_results
                }, f, indent=4)
            self.logger.info(f"Evaluation results saved to {results_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Classifier fitting or evaluation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def predict(self, feature_vector: np.ndarray) -> tuple:
        """
        Returns (predicted_character, confidence_score).
        Falls back to heuristic rules if no model has been trained yet.
        """
        if self._model is not None:
            try:
                feats = feature_vector.reshape(1, -1)
                
                # Check for label encoder compatibility
                if self.label_encoder is not None:
                    encoded_label = self._model.predict(feats)[0]
                    probs = self._model.predict_proba(feats)[0]
                    
                    # XGBoost or label encoder alignment check
                    if hasattr(self._model, "classes_"):
                        classes_arr = self._model.classes_
                        class_idx = np.where(classes_arr == encoded_label)[0][0]
                        confidence = float(probs[class_idx])
                    else:
                        confidence = float(np.max(probs))
                        
                    label = self.label_encoder.inverse_transform([encoded_label])[0]
                    return label, confidence
                else:
                    label = self._model.predict(feats)[0]
                    probs = self._model.predict_proba(feats)[0]
                    class_idx = np.where(self._model.classes_ == label)[0][0]
                    confidence = float(probs[class_idx])
                    return label, confidence
            except Exception as e:
                self.logger.error(f"Live prediction failed, falling back to heuristics: {e}")
                
        # Heuristic fallback rules (designed for static A, B, C, V, Y letters)
        # Using feature vector elements:
        # relative distances of tips (indices 63 to 67 correspond to Thumb, Index, Middle, Ring, Pinky tip-to-wrist)
        if len(feature_vector) >= 78:
            t_dist, i_dist, m_dist, r_dist, p_dist = feature_vector[63:68]
            i_to_t = feature_vector[68] # index-thumb tip distance
            
            # 1. Open palm - Letter 'B' (fingers straight)
            if i_dist >= 1.65 and m_dist >= 1.65 and r_dist >= 1.65 and p_dist >= 1.65:
                return "B", 0.85
                
            # 2. Two fingers up (peace sign) - Letter 'V'
            if i_dist >= 1.65 and m_dist >= 1.65 and r_dist < 1.3 and p_dist < 1.3:
                return "V", 0.80
                
            # 3. Pinky and Thumb extended - Letter 'Y'
            if t_dist >= 1.4 and p_dist >= 1.65 and i_dist < 1.3 and m_dist < 1.3 and r_dist < 1.3:
                return "Y", 0.85
                
            # 4. Closed fist - Letter 'A' (fingers curled, thumb tucked or slightly out)
            if i_dist < 1.3 and m_dist < 1.3 and r_dist < 1.3 and p_dist < 1.3:
                return "A", 0.80
                
            # 5. Cup hand - Letter 'C'
            # (Fingers partially bent but not curled completely, index-thumb distance is moderate)
            if (1.2 <= i_dist <= 1.65) and (1.2 <= m_dist <= 1.65) and (1.2 <= r_dist <= 1.65) and (0.4 <= i_to_t <= 1.25):
                return "C", 0.75
                
        return "None", 0.50
