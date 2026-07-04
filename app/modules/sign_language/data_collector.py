import os
import csv
import numpy as np
from app.core.interfaces import IStorageService

class SignLanguageDataCollector:
    def __init__(self, storage_service: IStorageService, csv_path="app/datasets/gestures_dataset.csv"):
        self._storage = storage_service
        self._csv_path = csv_path
        
        # Resolve absolute path for CSV
        self._resolve_csv_path()

    def _resolve_csv_path(self):
        if not os.path.isabs(self._csv_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self._csv_path = os.path.join(base_dir, self._csv_path)
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)

    def record_sample(self, label: str, landmarks: dict, version: str) -> bool:
        """
        Records a landmark sample to both SQLite and the CSV file.
        landmarks: dict of {0: np.array([x,y,z]), ... 20: np.array([x,y,z])}
        """
        if len(landmarks) < 21:
            print("[DataCollector Error] Cannot record sample: incomplete hand landmarks.")
            return False
            
        try:
            # 1. Save to SQLite database
            self._storage.save_landmarks(label, [landmarks[i] for i in range(21)], version)
            
            # 2. Save to CSV file
            write_headers = not os.path.exists(self._csv_path)
            
            flat_lms = []
            for i in range(21):
                flat_lms.extend(landmarks[i].tolist())
                
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_headers:
                    # Header row: label, x0, y0, z0, ..., x20, y20, z20, version
                    headers = ["label"]
                    for idx in range(21):
                        headers.extend([f"x{idx}", f"y{idx}", f"z{idx}"])
                    headers.append("version")
                    writer.writerow(headers)
                    
                row = [label] + flat_lms + [version]
                writer.writerow(row)
                
            return True
        except Exception as e:
            print(f"[DataCollector Error] Failed to write landmark sample: {e}")
            return False

    def export_csv_from_db(self, version: str) -> bool:
        """Exports SQLite database records for a version into the CSV file, replacing it."""
        try:
            records = self._storage.get_landmarks_dataset(version)
            if not records:
                print(f"[DataCollector Warning] No database records found for version {version}")
                return False
                
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["label"]
                for idx in range(21):
                    headers.extend([f"x{idx}", f"y{idx}", f"z{idx}"])
                headers.append("version")
                writer.writerow(headers)
                
                for row in records:
                    label = row[0]
                    coords = list(row[1:])
                    writer.writerow([label] + coords + [version])
                    
            print(f"[DataCollector] Exported {len(records)} samples to {self._csv_path}")
            return True
        except Exception as e:
            print(f"[DataCollector Error] Failed to export database records: {e}")
            return False
