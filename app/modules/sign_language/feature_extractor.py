import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, landmarks: dict) -> np.ndarray:
        """
        Extracts a flat feature vector from 21 landmarks.
        landmarks: dict {0: np.array([x,y,z]), 1: np.array([x,y,z]), ...}
        
        Returns a flat 1D numpy array of features (103 dimensions).
        """
        if len(landmarks) < 21:
            return np.zeros(103, dtype=np.float32)
            
        # 1. Translate coordinates so that the wrist (landmark 0) is at (0,0,0)
        wrist = landmarks[0]
        rel_coords = {}
        for idx in range(21):
            rel_coords[idx] = landmarks[idx] - wrist
            
        # 2. Scale normalization: calculate distance between wrist (0) and middle finger MCP (9)
        # to ensure the representation is scale-invariant.
        scale = np.linalg.norm(rel_coords[9])
        if scale < 1e-5:
            scale = 1.0
            
        norm_coords = {}
        for idx in range(21):
            norm_coords[idx] = rel_coords[idx] / scale
            
        # 3. Local Hand Coordinate System (to achieve rotation-invariance)
        # y-axis points from wrist to middle MCP (9)
        v_y = norm_coords[9]
        norm_vy = np.linalg.norm(v_y)
        y_basis = v_y / norm_vy if norm_vy > 1e-5 else np.array([0.0, 1.0, 0.0])
        
        # x-axis is perpendicular to y-axis in the plane of the palm (toward index MCP 5)
        v_5 = norm_coords[5]
        v_x = v_5 - np.dot(v_5, y_basis) * y_basis
        norm_vx = np.linalg.norm(v_x)
        x_basis = v_x / norm_vx if norm_vx > 1e-5 else np.array([1.0, 0.0, 0.0])
        
        # z-axis is the palm normal (orthogonal to x and y)
        z_basis = np.cross(x_basis, y_basis)
        norm_vz = np.linalg.norm(z_basis)
        if norm_vz > 1e-5:
            z_basis = z_basis / norm_vz
        else:
            z_basis = np.array([0.0, 0.0, 1.0])
            
        # Project all 21 normalized landmarks into the local coordinate system
        local_coords = {}
        for idx in range(21):
            local_coords[idx] = np.array([
                np.dot(norm_coords[idx], x_basis),
                np.dot(norm_coords[idx], y_basis),
                np.dot(norm_coords[idx], z_basis)
            ])
            
        # Flat list to accumulate all extracted features
        features = []
        
        # Features 1-63: Local rotation-invariant X, Y, Z of all 21 joints
        for idx in range(21):
            features.extend(local_coords[idx].tolist())
            
        # Features 64-68: Relative fingertip-to-wrist distances
        fingertips = [4, 8, 12, 16, 20]
        for tip in fingertips:
            dist = np.linalg.norm(local_coords[tip])
            features.append(float(dist))
            
        # Features 69-73: Adjacency fingertip-to-fingertip distances
        # (4-8, 8-12, 12-16, 16-20, 4-20)
        pair_distances = [
            (4, 8), (8, 12), (12, 16), (16, 20), (4, 20)
        ]
        for p1, p2 in pair_distances:
            dist = np.linalg.norm(local_coords[p1] - local_coords[p2])
            features.append(float(dist))
            
        # Features 74-78: Finger bending/flexion angles (PIP-to-TIP bend)
        # Calculated as angle between segment 1 (MCP-to-PIP) and segment 2 (PIP-to-TIP)
        finger_segments = [
            (2, 3, 4),     # Thumb
            (5, 6, 8),     # Index
            (9, 10, 12),   # Middle
            (13, 14, 16),  # Ring
            (17, 18, 20)   # Pinky
        ]
        
        def calculate_angle(v1, v2):
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-5 and norm2 > 1e-5:
                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            return 0.0
            
        for mcp, pip, tip in finger_segments:
            v1 = local_coords[pip] - local_coords[mcp]
            v2 = local_coords[tip] - local_coords[pip]
            features.append(calculate_angle(v1, v2))
            
        # --- NEW EXTENDED FEATURES FOR INCREASED ACCURACY ---
        
        # 1. Finger extension ratios (5 features)
        # Ratio of tip distance to MCP distance from wrist (0)
        mcps = [2, 5, 9, 13, 17]
        for i, tip in enumerate(fingertips):
            mcp = mcps[i]
            d_tip = np.linalg.norm(local_coords[tip])
            d_mcp = np.linalg.norm(local_coords[mcp])
            ratio = d_tip / d_mcp if d_mcp > 1e-5 else 0.0
            features.append(float(ratio))
            
        # 2. Detailed joint angles (14 features)
        # MCP angles (between wrist-to-MCP and MCP-to-PIP) and DIP/PIP angles
        extra_angles = [
            # Thumb: 1-2-3
            (local_coords[2]-local_coords[1], local_coords[3]-local_coords[2]),
            # Index: 0-5-6, 6-7-8
            (local_coords[5]-local_coords[0], local_coords[6]-local_coords[5]),
            (local_coords[7]-local_coords[6], local_coords[8]-local_coords[7]),
            # Middle: 0-9-10, 10-11-12
            (local_coords[9]-local_coords[0], local_coords[10]-local_coords[9]),
            (local_coords[11]-local_coords[10], local_coords[12]-local_coords[11]),
            # Ring: 0-13-14, 14-15-16
            (local_coords[13]-local_coords[0], local_coords[14]-local_coords[13]),
            (local_coords[15]-local_coords[14], local_coords[16]-local_coords[15]),
            # Pinky: 0-17-18, 18-19-20
            (local_coords[17]-local_coords[0], local_coords[18]-local_coords[17]),
            (local_coords[19]-local_coords[18], local_coords[20]-local_coords[19]),
            # Additional tip pairs to measure splay (5 pairs: 4-12, 4-16, 8-16, 8-20, 12-20)
            (local_coords[4]-local_coords[12], local_coords[8]-local_coords[16]),
            (local_coords[8]-local_coords[20], local_coords[12]-local_coords[20])
        ]
        
        # Flatten extra angles to keep list count matching 14 features
        for v1, v2 in extra_angles[:9]:
            features.append(calculate_angle(v1, v2))
        for p1, p2 in [(4, 12), (4, 16), (8, 16), (8, 20), (12, 20)]:
            features.append(float(np.linalg.norm(local_coords[p1] - local_coords[p2])))
            
        # 3. Palm orientation (3 features)
        # Components of the palm normal in standard camera coordinates
        features.extend(z_basis.tolist())
        
        # 4. Hand direction (3 features)
        # Components of the hand direction vector in standard camera coordinates
        features.extend(y_basis.tolist())
        
        return np.array(features, dtype=np.float32)
