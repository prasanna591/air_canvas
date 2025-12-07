import cv2
import numpy as np
import mediapipe as mp
import time
import os

class AirCanvas:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot access webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        _, frame = self.cap.read()
        self.height, self.width, _ = frame.shape

        # Canvas Setup
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_points = []
        self.current_color = (0, 0, 255)  # Default: Red
        self.brush_thickness = 5
        self.mode = "draw"  # Modes: draw, erase

        # Colors
        self.colors = {
            'Red': (0, 0, 255),
            'Blue': (255, 0, 0),
            'Green': (0, 255, 0),
            'Yellow': (0, 255, 255),
            'Purple': (255, 0, 255),
            'Orange': (0, 165, 255),
            'White': (255, 255, 255),
            'Black': (0, 0, 0),
            'Eraser': (0, 0, 0)
        }

        # Output Directory
        self.output_dir = "canvas_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # Frame Timing for FPS
        self.prev_time = time.time()
    
    def detect_hand_landmarks(self, image, hand_landmarks):
        """Draw hand landmarks"""
        for hand_lms in hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
    
    def get_finger_positions(self, hand_landmarks):
        """Extract finger positions"""
        if not hand_landmarks:
            return None, None, None
        
        hand_lms = hand_landmarks[0]
        index_tip = (int(hand_lms.landmark[8].x * self.width), int(hand_lms.landmark[8].y * self.height))
        index_mcp = (int(hand_lms.landmark[5].x * self.width), int(hand_lms.landmark[5].y * self.height))
        middle_tip = (int(hand_lms.landmark[12].x * self.width), int(hand_lms.landmark[12].y * self.height))
        
        return index_tip, index_mcp, middle_tip
    
    def is_drawing_gesture(self, index_tip, index_mcp, middle_tip):
        """Check for drawing gesture"""
        index_extended = index_tip[1] < index_mcp[1] - 40
        middle_distance = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))
        return index_extended and middle_distance > 80
    
    def is_eraser_gesture(self, index_tip, index_mcp, middle_tip):
        """Check for eraser gesture"""
        index_extended = index_tip[1] < index_mcp[1] - 40
        middle_extended = middle_tip[1] < index_mcp[1] - 40
        return index_extended and middle_extended
    
    def draw_on_canvas(self, point):
        """Draw or erase on canvas"""
        if self.mode == "draw":
            if len(self.drawing_points) < 2:
                self.drawing_points.append(point)
            else:
                self.drawing_points.append(point)
                cv2.line(self.canvas, self.drawing_points[-2], self.drawing_points[-1], self.current_color, self.brush_thickness)
        elif self.mode == "erase":
            cv2.circle(self.canvas, point, 30, (0, 0, 0), -1)
    
    def draw_ui(self, frame):
        """Display UI Elements"""
        # Draw FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - self.prev_time))
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mode and Color
        cv2.putText(frame, f"Mode: {self.mode.capitalize()} | Color: {list(self.colors.keys())[list(self.colors.values()).index(self.current_color)]}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw Color Palette
        y_pos = 150
        for color_name, color_value in self.colors.items():
            cv2.circle(frame, (30, y_pos), 15, color_value, -1)
            if color_value == self.current_color:
                cv2.circle(frame, (30, y_pos), 18, (255, 255, 255), 2)
            y_pos += 40
    
    def save_canvas(self):
        """Save the canvas as an image"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.output_dir, f"canvas_{timestamp}.png")
        cv2.imwrite(filename, self.canvas)
        print(f"Canvas saved as {filename}")

    def run(self):
        with self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    self.detect_hand_landmarks(frame, results.multi_hand_landmarks)
                    index_tip, index_mcp, middle_tip = self.get_finger_positions(results.multi_hand_landmarks)

                    if self.is_eraser_gesture(index_tip, index_mcp, middle_tip):
                        self.mode = "erase"
                        self.draw_on_canvas(index_tip)
                    elif self.is_drawing_gesture(index_tip, index_mcp, middle_tip):
                        self.mode = "draw"
                        self.draw_on_canvas(index_tip)
                    else:
                        self.drawing_points = []
                
                combined_frame = cv2.addWeighted(frame, 1.0, self.canvas, 0.7, 0)
                self.draw_ui(combined_frame)
                cv2.imshow('Air Canvas', combined_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.canvas.fill(0)
                elif key == ord('s'):
                    self.save_canvas()
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AirCanvas()
    app.run()
