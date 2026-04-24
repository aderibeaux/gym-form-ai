#so this is from chat
#Your script should:
#1. import OpenCV and MediaPipe
#2. create a MediaPipe pose object
#3. open the webcam
#4. read frames in a loop
#5. convert each frame from BGR to RGB
#6. run pose detection on the frame
#7. draw the pose landmarks on the frame
#8. show the frame in a window
#9. stop when you press a key like q
#10. release the webcam and close windows

#FOR REFERENCE ON MEDIAPOSE USE THIS LINK: 
#https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md

#now this is me
#1
import cv2
import mediapipe as mp
import pandas as pd

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#2 most of this i got from the linke above (in the mediapipe documentation)
#made 2 small modifications - one for camera being on and the other just in case
#there is not body detected

#MAY WANT TO COME BACK AND LOOK AT THE CAMERA FLIPPED THIN
#unsure if I am going to make adjustments but for now it is same as OG


cap = cv2.VideoCapture(1)
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



