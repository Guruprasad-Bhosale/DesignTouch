import os
import sys
import sqlite3
import numpy as np
import random

# Adjust path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from app.services.storage_service import StorageService

def get_rotation_matrix(roll, pitch, yaw):
    # R_x (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    # R_y (yaw)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    # R_z (roll)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def generate_base_hand(pose_name):
    # Base scale
    S = 0.15
    L_finger = 0.08
    L_thumb = 0.06
    
    # 21 landmarks
    lms = {}
    lms[0] = np.array([0.0, 0.0, 0.0]) # Wrist
    
    # Base MCP joints (relative to wrist)
    lms[1] = np.array([-0.02 * S, -0.04 * S, 0.0]) # Thumb CMC
    lms[2] = np.array([-0.04 * S, -0.08 * S, 0.0]) # Thumb MCP
    lms[5] = np.array([-0.03 * S, -0.14 * S, 0.0]) # Index MCP
    lms[9] = np.array([0.0 * S, -0.15 * S, 0.0])   # Middle MCP
    lms[13] = np.array([0.03 * S, -0.14 * S, 0.0]) # Ring MCP
    lms[17] = np.array([0.05 * S, -0.12 * S, 0.0]) # Pinky MCP
    
    # Finger configs for each letter
    # Format: (Index, Middle, Ring, Pinky, Thumb)
    # E=extended, C=curled, P=partially_curled, X=crossed, H=hooked
    # Thumb: E=extended, C=curled_across_palm, F=folded_over_fingers
    cfg = {
        "A": ("C", "C", "C", "C", "A_THUMB"),
        "B": ("E", "E", "E", "E", "C"),
        "C": ("P", "P", "P", "P", "C_THUMB"),
        "D": ("E", "C", "C", "C", "D_THUMB"),
        "E": ("C", "C", "C", "C", "E_THUMB"),
        "F": ("F_INDEX", "E", "E", "E", "F_THUMB"), # index and thumb touch
        "G": ("E", "C", "C", "C", "E"), 
        "H": ("E", "E", "C", "C", "C"), 
        "I": ("C", "C", "C", "E", "C"), 
        "J": ("C", "C", "C", "E", "C"), # J has different rotation offset applied later
        "K": ("E", "E", "C", "C", "K_THUMB"), 
        "L": ("E", "C", "C", "C", "E"), 
        "M": ("C", "C", "C", "C", "M_THUMB"), 
        "N": ("C", "C", "C", "C", "N_THUMB"), 
        "O": ("O_CURVE", "O_CURVE", "O_CURVE", "O_CURVE", "O_THUMB"), 
        "P": ("E", "E", "C", "C", "K_THUMB"), 
        "Q": ("E", "C", "C", "C", "E"), 
        "R": ("X", "X", "C", "C", "C"), 
        "S": ("C", "C", "C", "C", "S_THUMB"), 
        "T": ("C", "C", "C", "C", "T_THUMB"), 
        "U": ("E", "E", "C", "C", "C"), 
        "V": ("E", "E", "C", "C", "C"), 
        "W": ("E", "E", "E", "C", "C"), 
        "X": ("H", "C", "C", "C", "C"), 
        "Y": ("C", "C", "C", "E", "E"), 
        "Z": ("E", "C", "C", "C", "C")  
    }
    
    pose = cfg.get(pose_name, ("E", "E", "E", "E", "E"))
    
    # 1. Fingers (Index, Middle, Ring, Pinky)
    finger_mcps = [5, 9, 13, 17]
    finger_dirs = [
        np.array([-0.05, -1.0, 0.0]), # Index
        np.array([0.0, -1.0, 0.0]),   # Middle
        np.array([0.05, -1.0, 0.0]),  # Ring
        np.array([0.1, -1.0, 0.0])    # Pinky
    ]
    # Splay
    if pose_name in ["U", "R"]:
        finger_dirs[0] = np.array([-0.01, -1.0, 0.0])
        finger_dirs[1] = np.array([0.01, -1.0, 0.0])
    elif pose_name == "V":
        finger_dirs[0] = np.array([-0.12, -1.0, 0.0])
        finger_dirs[1] = np.array([0.12, -1.0, 0.0])
    elif pose_name == "W":
        finger_dirs[0] = np.array([-0.18, -1.0, 0.0])
        finger_dirs[1] = np.array([0.0, -1.0, 0.0])
        finger_dirs[2] = np.array([0.18, -1.0, 0.0])
        
    for i, mcp in enumerate(finger_mcps):
        f_dir = finger_dirs[i]
        f_dir = f_dir / np.linalg.norm(f_dir)
        p_state = pose[i]
        
        mcp_pt = lms[mcp]
        if p_state == "E": # Extended
            lms[mcp+1] = mcp_pt + f_dir * L_finger * 0.4
            lms[mcp+2] = lms[mcp+1] + f_dir * L_finger * 0.3
            lms[mcp+3] = lms[mcp+2] + f_dir * L_finger * 0.3
        elif p_state in ["C", "F_INDEX"]: # Curled
            lms[mcp+1] = mcp_pt + f_dir * L_finger * 0.4
            lms[mcp+2] = lms[mcp+1] + np.array([0.0, -0.1, 0.6]) * L_finger * 0.3
            lms[mcp+3] = lms[mcp+2] - f_dir * L_finger * 0.3
        elif p_state == "P": # Partially curled (C shape)
            lms[mcp+1] = mcp_pt + (f_dir + np.array([0.0, 0.0, 0.3])) * L_finger * 0.4
            lms[mcp+2] = lms[mcp+1] + (f_dir * 0.3 + np.array([0.0, 0.0, 0.6])) * L_finger * 0.3
            lms[mcp+3] = lms[mcp+2] + (np.array([0.0, 0.3, 0.4])) * L_finger * 0.3
        elif p_state == "O_CURVE": # O circle curve
            lms[mcp+1] = mcp_pt + (f_dir + np.array([0.0, 0.0, 0.5])) * L_finger * 0.4
            lms[mcp+2] = lms[mcp+1] + (f_dir * 0.1 + np.array([0.0, 0.0, 0.8])) * L_finger * 0.3
            lms[mcp+3] = lms[mcp+2] + (np.array([0.0, 0.5, 0.3])) * L_finger * 0.3
        elif p_state == "X": # Crossed (R)
            if mcp == 5:
                cross_dir = np.array([0.06, -1.0, 0.02])
            else:
                cross_dir = np.array([-0.06, -1.0, -0.02])
            cross_dir = cross_dir / np.linalg.norm(cross_dir)
            lms[mcp+1] = mcp_pt + cross_dir * L_finger * 0.4
            lms[mcp+2] = lms[mcp+1] + cross_dir * L_finger * 0.3
            lms[mcp+3] = lms[mcp+2] + cross_dir * L_finger * 0.3
        elif p_state == "H": # Hooked (X)
            lms[mcp+1] = mcp_pt + f_dir * L_finger * 0.4
            lms[mcp+2] = lms[mcp+1] + np.array([0.0, 0.4, 0.6]) * L_finger * 0.3
            lms[mcp+3] = lms[mcp+2] + np.array([0.0, 0.6, 0.2]) * L_finger * 0.3
            
    # 2. Thumb
    t_state = pose[4]
    mcp_t = lms[2]
    
    if t_state == "E": # Extended out
        t_dir = np.array([-0.8, -0.4, -0.1])
        t_dir = t_dir / np.linalg.norm(t_dir)
        lms[3] = mcp_t + t_dir * L_thumb * 0.5
        lms[4] = lms[3] + t_dir * L_thumb * 0.5
    elif t_state == "C": # Curled across palm
        t_dir = np.array([0.7, -0.1, 0.3])
        t_dir = t_dir / np.linalg.norm(t_dir)
        lms[3] = mcp_t + t_dir * L_thumb * 0.5
        lms[4] = lms[3] + t_dir * L_thumb * 0.5
    elif t_state == "A_THUMB": # A: thumb resting on the side of index finger
        lms[3] = lms[5] + np.array([-0.02 * S, 0.0, 0.05 * S])
        lms[4] = lms[5] + np.array([-0.02 * S, 0.04 * S, 0.05 * S])
    elif t_state == "E_THUMB": # E: thumb bent in front of curled fingers
        lms[3] = np.array([-0.02 * S, -0.11 * S, 0.06 * S])
        lms[4] = np.array([0.02 * S, -0.11 * S, 0.06 * S])
    elif t_state == "M_THUMB": # M: thumb tucked under index, middle, ring
        # Placed near ring finger MCP/PIP
        lms[3] = lms[13] + np.array([0.0, 0.02 * S, 0.05 * S])
        lms[4] = lms[13] + np.array([0.0, 0.05 * S, 0.05 * S])
    elif t_state == "N_THUMB": # N: thumb tucked under index and middle
        # Placed near middle finger MCP/PIP
        lms[3] = lms[9] + np.array([0.0, 0.02 * S, 0.05 * S])
        lms[4] = lms[9] + np.array([0.0, 0.05 * S, 0.05 * S])
    elif t_state == "T_THUMB": # T: thumb tucked under index
        # Placed near index finger MCP/PIP
        lms[3] = lms[5] + np.array([0.0, 0.02 * S, 0.05 * S])
        lms[4] = lms[5] + np.array([0.0, 0.05 * S, 0.05 * S])
    elif t_state == "S_THUMB": # S: thumb wrapped over index/middle fingers
        lms[3] = np.array([-0.01 * S, -0.12 * S, 0.08 * S])
        lms[4] = np.array([0.02 * S, -0.12 * S, 0.08 * S])
    elif t_state == "D_THUMB": # D: thumb touching middle finger tip
        lms[3] = lms[10] + np.array([-0.01 * S, 0.0, 0.03 * S])
        lms[4] = lms[12]  # touches middle fingertip
    elif t_state == "C_THUMB": # C: curved thumb
        t_dir = np.array([-0.5, -0.4, 0.5])
        t_dir = t_dir / np.linalg.norm(t_dir)
        lms[3] = mcp_t + t_dir * L_thumb * 0.5
        lms[4] = lms[3] + np.array([0.3, 0.3, 0.5]) * L_thumb * 0.5
    elif t_state == "O_THUMB": # O: thumb touches tips of fingers
        lms[3] = np.array([-0.01 * S, -0.1 * S, 0.08 * S])
        lms[4] = np.array([0.0, -0.09 * S, 0.08 * S])
        # Force other fingertips to touch thumb tip
        for mcp in [5, 9, 13, 17]:
            lms[mcp+3] = lms[4]
    elif t_state == "F_THUMB": # F: thumb touching index fingertip
        lms[3] = lms[2] + np.array([0.01 * S, -0.04 * S, 0.05 * S])
        lms[4] = lms[8] # touches index fingertip
        lms[8] = lms[8] + np.array([0.0, 0.0, 0.02 * S]) # bring them closer
    elif t_state == "K_THUMB": # K: thumb touches middle finger knuckle
        lms[3] = lms[2] + np.array([0.01 * S, -0.03 * S, 0.04 * S])
        lms[4] = lms[10] # touches middle PIP (knuckle)

    # Standardize scale: ensure wrist to middle MCP (9) is exactly 0.15
    scale_act = np.linalg.norm(lms[9] - lms[0])
    if scale_act > 0:
        factor = 0.15 / scale_act
        for idx in range(21):
            lms[idx] = lms[idx] * factor
            
    return lms

def generate_dataset(num_samples_per_char=100, version="v1.0"):
    storage = StorageService()
    
    conn = sqlite3.connect(storage._db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM landmarks_dataset WHERE version = ?", (version,))
    conn.commit()
    conn.close()
    
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total_samples = 0
    
    for char in letters:
        for sample_idx in range(num_samples_per_char):
            lms = generate_base_hand(char)
            
            # Apply 3D rotation
            roll = random.uniform(-0.10, 0.10)
            pitch = random.uniform(-0.10, 0.10)
            yaw = random.uniform(-0.10, 0.10)
            
            # Rotation offsets for specific signs
            if char == "P":
                pitch += 0.8
            elif char == "Q":
                pitch += 1.0
            elif char in ["G", "H"]:
                yaw += 0.5
                roll -= 0.5
            elif char == "J":
                # J has starting tilt offset to distinguish from I statically
                pitch += 0.2
                yaw += 0.2
            elif char == "Z":
                # Z has starting tilt offset to distinguish from D statically
                pitch -= 0.2
                roll += 0.2
                
            R = get_rotation_matrix(roll, pitch, yaw)
            scale_fac = random.gauss(1.0, 0.05)
            
            shift_x = random.gauss(0.5, 0.02)
            shift_y = random.gauss(0.6, 0.02)
            shift_z = random.gauss(0.0, 0.01)
            
            transformed_lms = []
            for idx in range(21):
                pt = lms[idx]
                pt_rot = (R @ pt) * scale_fac
                pt_trans = pt_rot + np.array([shift_x, shift_y, shift_z])
                # Add Gaussian noise
                noise = np.random.normal(0, 0.0008, 3)
                pt_noisy = pt_trans + noise
                transformed_lms.append(pt_noisy)
                
            storage.save_landmarks(char, transformed_lms, version)
            total_samples += 1
            
    print(f"Successfully generated refined {total_samples} samples in dataset version '{version}'.")

if __name__ == "__main__":
    generate_dataset()
