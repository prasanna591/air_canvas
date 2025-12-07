import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utility
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define a function to detect hand landmarks and draw on canvas
def detect_hand_landmarks(image, hand_landmarks):
    if hand_landmarks:
        for handLms in hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:  # Index finger tip
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

# Define a function to draw on the canvas
def draw_on_canvas(points, canvas, color):
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(canvas, points[i - 1], points[i], color, 5)

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Initialize variables for drawing
drawing_points = []
current_color = (0, 0, 255)  # Default color is red

# Start MediaPipe Hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a natural selfie-view display
        frame = cv2.flip(frame, 1)

        # Get the frame dimensions
        h, w, _ = frame.shape

        # Create a blank canvas matching the frame dimensions
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert the BGR frame to RGB before processing with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe to detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            detect_hand_landmarks(frame, results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmark for the index finger tip (id 8)
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                drawing_points.append((x, y))
        else:
            drawing_points.append(None)

        # Draw on canvas
        draw_on_canvas(drawing_points, canvas, current_color)

        # Merge the frame and canvas
        combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Display the final result
        cv2.imshow('Air Canvas', combined_frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset canvas
            drawing_points = []
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        elif key == ord('b'):
            current_color = (255, 0, 0)  # Blue
        elif key == ord('g'):
            current_color = (0, 255, 0)  # Green
        elif key == ord('y'):
            current_color = (0, 255, 255)  # Yellow
        elif key == ord('w'):
            current_color = (255, 255, 255)  # White

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
