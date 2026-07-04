import os
import sys
import numpy as np
import json
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.services.storage_service import StorageService
from app.modules.sign_language.feature_extractor import FeatureExtractor
from app.modules.sign_language.classifier import SignLanguageClassifier

def run_audit(dataset_version="v1.0"):
    print("=== STARTING SIGN LANGUAGE PIPELINE AUDIT ===")
    
    storage = StorageService()
    extractor = FeatureExtractor()
    classifier = SignLanguageClassifier()
    
    # 1. Load landmarks from database
    records = storage.get_landmarks_dataset(dataset_version)
    if not records:
        print(f"[Error] No records found in database for version {dataset_version}")
        return
        
    print(f"Loaded {len(records)} samples from version '{dataset_version}'")
    
    features_list = []
    labels = []
    
    for row in records:
        label = row[0]
        coords = list(row[1:])
        
        # Reconstruct landmarks mapping
        lms_dict = {}
        for idx in range(21):
            x = coords[idx * 3]
            y = coords[idx * 3 + 1]
            z = coords[idx * 3 + 2]
            lms_dict[idx] = np.array([x, y, z])
            
        features = extractor.extract_features(lms_dict)
        features_list.append(features)
        labels.append(label)
        
    X = np.array(features_list)
    y = labels
    
    # 2. Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Train the classifier on train split
    print("Training candidate models and selecting the best...")
    success = classifier.train(X_train, y_train)
    if not success:
        print("[Error] Failed to train classifier.")
        return
        
    # Load best model back
    classifier.load()
    
    # 4. Predict on validation split
    y_pred = []
    y_conf = []
    
    for feat in X_val:
        label, conf = classifier.predict(feat)
        y_pred.append(label)
        y_conf.append(conf)
        
    # 5. Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)
    classes = sorted(list(set(y_val)))
    
    # Identify commonly confused letters
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                confusions.append({
                    "true_class": classes[i],
                    "predicted_class": classes[j],
                    "count": int(cm[i][j])
                })
                
    # Sort confusions by count descending
    confusions = sorted(confusions, key=lambda x: x["count"], reverse=True)
    
    # Write diagnostics to files
    diagnostics = {
        "accuracy": float(acc),
        "dataset_version": dataset_version,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "common_confusions": confusions
    }
    
    diag_path = "app/models/diagnostics_report.json"
    abs_diag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), diag_path)
    os.makedirs(os.path.dirname(abs_diag_path), exist_ok=True)
    
    with open(abs_diag_path, "w") as f:
        json.dump(diagnostics, f, indent=4)
        
    # Print markdown report summary
    report_md_path = abs_diag_path.replace(".json", ".md")
    with open(report_md_path, "w") as f:
        f.write("# Sign Language Pipeline Diagnostics Report\n\n")
        f.write(f"- **Dataset Version**: {dataset_version}\n")
        f.write(f"- **Total Validation Samples**: {len(y_val)}\n")
        f.write(f"- **Global Accuracy**: {acc*100:.2f}%\n\n")
        
        f.write("## Per-Class Evaluation Metrics\n\n")
        f.write("| Letter | Precision | Recall | F1-Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        for cls in classes:
            metrics = report[cls]
            f.write(f"| {cls} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1-score']:.3f} | {metrics['support']} |\n")
            
        f.write("\n## Commonly Confused Letters\n\n")
        if confusions:
            f.write("| True Letter | Predicted as | Confused Count |\n")
            f.write("|---|---|---|\n")
            for conf in confusions[:10]: # Top 10 confusions
                f.write(f"| {conf['true_class']} | {conf['predicted_class']} | {conf['count']} |\n")
        else:
            f.write("No confusions detected! All validation letters recognized perfectly.\n")
            
    print(f"Diagnostics JSON saved to: {abs_diag_path}")
    print(f"Diagnostics Markdown Report saved to: {report_md_path}")
    print("\n--- Commonly Confused Letters (Top 5) ---")
    for conf in confusions[:5]:
        print(f"True '{conf['true_class']}' predicted as '{conf['predicted_class']}' ({conf['count']} times)")

if __name__ == "__main__":
    run_audit()
