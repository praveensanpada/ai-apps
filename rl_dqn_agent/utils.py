# === utils.py ===
import numpy as np

def epsilon_by_frame(frame_idx, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)
