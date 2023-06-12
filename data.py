import cv2
import json
import mediapipe as mp

# Load the pre-trained MediaPipe Pose model
mp_pose = mp.solutions.pose

# Load the video file
video_path = r"C:\Users\prajw\OneDrive\Desktop\VSCode\Python\Summer_Internship\Ballet.mp4"

cap = cv2.VideoCapture(video_path)

# Create an instance of the MediaPipe Pose model
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
    pose_data = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose
        results = pose.process(image_rgb)

        # Store skeletal joint positions in a list
        frame_joints = []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                frame_joints.append((idx, landmark.x, landmark.y, landmark.z))
            pose_data.append(frame_joints)

        # Display the output
        cv2.imshow("MediaPipe Skeletal Conversion", image)
        if cv2.waitKey(1) == ord('q'):
            break

    # Save the pose data as JSON
    with open("skeletal_data.json", "w") as f:
        json.dump(pose_data, f)

cap.release()
cv2.destroyAllWindows()
