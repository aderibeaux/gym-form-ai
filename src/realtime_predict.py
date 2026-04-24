import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from scipy.signal import find_peaks

#most of this is copy paste what i did previously
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

model = joblib.load("models/squat_model.pkl")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#2 most of this i got from the linke above (in the mediapipe documentation)
#made 2 small modifications - one for camera being on and the other just in case
#there is not body detected

#MAY WANT TO COME BACK AND LOOK AT THE CAMERA FLIPPED THIN
#unsure if I am going to make adjustments but for now it is same as OG


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Couldn't open webcam.")
    exit()
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  #add this little safety check so i know the problem is that the camera isn't open
  rows = []
  frame_count = 0

  while cap.isOpened():
    frame_count += 1
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #adjusted this from original because I only want to call that if a body is detetcted
    if results.pose_landmarks:
      landmarks = results.pose_landmarks.landmark
      shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
      hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
      knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
      ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
      rows.append([frame_count, shoulder.x, shoulder.y, shoulder.visibility,
                  hip.x, hip.y, hip.visibility,
                  knee.x, knee.y, knee.visibility,
                  ankle.x, ankle.y, ankle.visibility])
      mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
df = pd.DataFrame(rows, columns = ["frame", "shoulder_x", "shoulder_y", "shoulder_visibility",
                                   "hip_x", "hip_y", "hip_visibility", "knee_x", "knee_y", "knee_visibility", 
                                   "ankle_x", "ankle_y", "ankle_visibility"])
df.to_csv("data/pose_landmarks.csv", index=False)
cap.release()
cv2.destroyAllWindows() #chat said to do this to close the opencv display windows

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

feature_cols = [
   "rep_duration_frames",
    "hip_range",
    "bottom_knee_angle",
    "bottom_hip_angle",
    "bottom_torso_angle",
    "bottom_shin_angle"
    ]

X_new = features_df[feature_cols]

predictions = model.predict(X_new)
features_df["predicted_label"] = predictions

def feedback(label):
   if label == "good":
      return "Good Form!"
   elif label == "shallow":
      return "Try going deeper (knee should be 90 degrees)"
   elif label == "upright":
      return "Lean torso forward slightly"
   else:
      return "Unknown"


features_df["feedback"] = features_df["predicted_label"].apply(feedback)
features_df.to_csv("data/predicted_reps.csv", index=False)
features_df.to_csv("data/predicted_reps_with_feedback.csv", index=False)

print("Saved predicted_reps_with_feedback.csv")

print("\nPredictions:")
print(features_df[["rep_num", "predicted_label", "feedback"]])

print("\nSaved:")
print("new_pose_landmarks.csv")
print("predicted_reps.csv")
