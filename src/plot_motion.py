import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a-b
    bc = c-b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba ==0 or norm_bc ==0:
        return np.nan
    
    cosine_angle = np.dot(ba,bc)/(norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def angle_to_vertical(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    vector = p2-p1
    vertical = np.array([0,1])
    norm_vector = np.linalg.norm(vector)
    norm_vertical = np.linalg.norm(vertical)

    if norm_vector ==0 or norm_vertical == 0:
        return np.nan
    
    cosine_angle = np.dot(vector, vertical)/(norm_vector * norm_vertical)
    cosine_angle = np.clip(cosine_angle, -1.0,1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    
    if angle >90:
        angle = 180-angle
    return angle

df = pd.read_csv("data/pose_landmarks.csv")
#used AI for the smoothing
df["hip_y_smooth"] = df["hip_y"].rolling(window = 5, center = True).mean()
signal = df["hip_y_smooth"].dropna()

peaks, _ = find_peaks(signal, distance = 40, prominence=0.05)

signal_index = signal.index[peaks]

bottom_indices = signal.index[peaks].to_list()
bottom_indices = sorted(bottom_indices)

rep_intervals = []

if len(bottom_indices) >= 1:
    boundaries = [0]
    for i in range (len(bottom_indices)-1):
        midpoint = (bottom_indices[i] + bottom_indices[i+1])//2
        boundaries.append(midpoint)
    boundaries.append(len(df) - 1)

    for i, bottom_idx in enumerate(bottom_indices):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]

        rep_intervals.append({
            "rep_num": i+1,
            "start_idx": start_idx,
            "bottom_idx": bottom_idx,
            "end_idx": end_idx
        })

        print(rep_intervals)


feature_rows = []

for rep in rep_intervals:
    start_idx = rep["start_idx"]
    end_idx = rep["end_idx"]
    rep_num = rep["rep_num"]

    rep_df = df.iloc[start_idx:end_idx+1]
    bottom_idx = rep["bottom_idx"]
    bottom_row = df.loc[bottom_idx]

    shoulder = (bottom_row["shoulder_x"], bottom_row["shoulder_y"])
    hip = (bottom_row["hip_x"], bottom_row["hip_y"])
    knee = (bottom_row["knee_x"], bottom_row["knee_y"])
    ankle = (bottom_row["ankle_x"], bottom_row["ankle_y"])
    
    #ANGLESSSS
    bottom_knee_angle = calculate_angle(hip, knee, ankle)
    bottom_hip_angle = calculate_angle(shoulder, hip, knee)
    
    bottom_torso_angle = angle_to_vertical(hip, shoulder)
    bottom_shin_angle = angle_to_vertical(ankle, knee)


    rep_duration_frames = len(rep_df)
    hip_range = rep_df["hip_y"].max() - rep_df["hip_y"].min()
    
    row = {
        "rep_num": rep_num,
        "start_idx": start_idx,
        "bottom_idx": rep["bottom_idx"],
        "end_idx": end_idx,
        "rep_duration_frames": rep_duration_frames,
        "hip_range": hip_range,
        "bottom_knee_angle": bottom_knee_angle,
        "bottom_hip_angle": bottom_hip_angle,
        "bottom_torso_angle": bottom_torso_angle,
        "bottom_shin_angle": bottom_shin_angle
    }

    feature_rows.append(row)

features_df = pd.DataFrame(feature_rows)

print("\n Rep features")
print(features_df)

features_df.to_csv("data/features.csv", index=False)

plt.plot(df["frame"], df["hip_y_smooth"])
plt.plot(df.loc[signal_index, "frame"], df.loc[signal_index, "hip_y_smooth"], "ro", label="Detected rep bottoms")
plt.xlabel("Frame")
plt.ylabel("Hip Y Position")
plt.title("Bulgarian Split Squat Hip Movement Over Time")
plt.show()

