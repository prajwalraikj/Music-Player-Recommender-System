import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# Load video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Detection
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(image_rgb)

        # Draw bounding boxes on the image
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Display the output
        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
