import cv2
import json
import os
import mediapipe as mp

# Load the pre-trained MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load the video file
video_path = r"C:\Users\prajw\OneDrive\Desktop\VSCode\Python\Summer_Internship\Dance\HipHop.mp4"

cap = cv2.VideoCapture(video_path)

# Get the original video's frame count and duration
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frame_count)
# duration = frame_count / cap.get(cv2.CAP_PROP_FPS)
# print(duration)
# print(cap.get(cv2.CAP_PROP_FPS))
# Create an instance of the MediaPipe Pose model
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
    pose_data = []

    # for frame_index in range(frame_count):
    #     # Set the current frame position
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    #     # Read the frame
    #     success, image = cap.read()
    #     if not success:
    #         break
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break


        # Process the image with MediaPipe Pose
        results = pose.process(image)

        # Store skeletal joint positions in a list
        frame_joints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                frame_joints.append((landmark.x, landmark.y, landmark.z))
            pose_data.append(frame_joints)

        # Draw skeletal landmarks on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the output
        cv2.imshow("MediaPipe Skeletal Conversion", image)
        # cv2.waitKey(1)

        if cv2.waitKey(1) == ord('q'):
            break

    # Create the 'Datasets' folder if it does not exist
    dataset_folder = "Datasets"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Save the pose data as JSON in the 'Datasets' folder
    json_file_path = os.path.join(dataset_folder, "skeletal_data.json")
    with open(json_file_path, "w") as f:
        json.dump(pose_data, f)

cap.release()
cv2.destroyAllWindows()
