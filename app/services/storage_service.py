import sqlite3
import os
import json
import numpy as np
from app.core.interfaces import IStorageService

class StorageService(IStorageService):
    def __init__(self, db_path="app/datasets/gesture_verse.db"):
        self._db_path = db_path
        self.initialize()

    def initialize(self):
        # Resolve absolute path if needed
        if not os.path.isabs(self._db_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._db_path = os.path.join(base_dir, self._db_path)
            
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        print(f"[StorageService] Connecting to SQLite database at {self._db_path}")
        
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        # 1. Create settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # 2. Create landmarks_dataset table
        # Columns: id, label, x0, y0, z0, ... x20, y20, z20, version
        coords_cols = []
        for i in range(21):
            coords_cols.append(f"x{i} REAL")
            coords_cols.append(f"y{i} REAL")
            coords_cols.append(f"z{i} REAL")
            
        cols_query = ", ".join(coords_cols)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS landmarks_dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                {cols_query},
                version TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        print("[StorageService] SQLite Schema verified and initialized.")

    def get_setting(self, key: str, default=None):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        
        if row is not None:
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]
        return default

    def set_setting(self, key: str, value):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        val_str = json.dumps(value)
        cursor.execute("""
            INSERT OR REPLACE INTO settings (key, value)
            VALUES (?, ?)
        """, (key, val_str))
        conn.commit()
        conn.close()

    def save_landmarks(self, label: str, landmarks: list, version: str):
        """
        landmarks is a list of 21 landmarks, each being an object or dict/array with .x, .y, .z
        or array/list of 3 values.
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        # Flatten landmarks into a list of 63 floats
        flat_coords = []
        for lm in landmarks:
            if hasattr(lm, 'x'):
                flat_coords.extend([float(lm.x), float(lm.y), float(lm.z)])
            elif isinstance(lm, (list, tuple, np.ndarray)):
                flat_coords.extend([float(lm[0]), float(lm[1]), float(lm[2])])
            else:
                raise ValueError("Landmarks elements must have x, y, z properties or support indexing")
                
        # Build query placeholders
        placeholders = ", ".join(["?"] * 63)
        cols = ", ".join([f"x{i}, y{i}, z{i}" for i in range(21)])
        
        query = f"""
            INSERT INTO landmarks_dataset (label, {cols}, version)
            VALUES (?, {placeholders}, ?)
        """
        
        cursor.execute(query, [label] + flat_coords + [version])
        conn.commit()
        conn.close()

    def get_landmarks_dataset(self, version: str) -> list:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        cols = ", ".join([f"x{i}, y{i}, z{i}" for i in range(21)])
        query = f"SELECT label, {cols} FROM landmarks_dataset WHERE version = ?"
        
        cursor.execute(query, (version,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_dataset_versions(self) -> list:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT version FROM landmarks_dataset")
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]
