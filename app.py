import cv2
import numpy as np
import mediapipe as mp
import time
import os

class AirCanvas:
    def __init__(self):
        # Initialize MediaPipe Hands and Drawing utility
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot access webcam")
        
        # Set resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get frame dimensions
        _, frame = self.cap.read()
        self.height, self.width, _ = frame.shape
        
        # Drawing parameters
        self.drawing_points = []
        self.current_color = (0, 0, 255)  # Default color is red
        self.brush_thickness = 5
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Mode settings
        self.mode = "draw"  # Modes: "draw", "erase"
        
        # Tracking parameters
        self.is_drawing = False
        self.prev_time = 0
        self.curr_time = 0
        
        # Color palette (BGR format)
        self.colors = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'orange': (0, 165, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'eraser': (0, 0, 0)  # Eraser is black (same as background)
        }
        
        # Create output directory for saved images
        self.output_dir = "canvas_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def detect_hand_landmarks(self, image, hand_landmarks):
        """Detect and visualize hand landmarks"""
        if hand_landmarks:
            for hand_lms in hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
    
    def get_finger_positions(self, hand_landmarks):
        """Extract key finger positions for controls"""
        if not hand_landmarks:
            return None, None, None
        
        hand_lms = hand_landmarks[0]  # Get first hand
        
        # Get index finger tip position (for drawing)
        index_tip = (
            int(hand_lms.landmark[8].x * self.width),
            int(hand_lms.landmark[8].y * self.height)
        )
        
        # Get index finger MCP (knuckle) position
        index_mcp = (
            int(hand_lms.landmark[5].x * self.width),
            int(hand_lms.landmark[5].y * self.height)
        )
        
        # Get middle finger tip position (for mode control)
        middle_tip = (
            int(hand_lms.landmark[12].x * self.width),
            int(hand_lms.landmark[12].y * self.height)
        )
        
        return index_tip, index_mcp, middle_tip
    
    def is_drawing_gesture(self, index_tip, index_mcp, middle_tip):
        """Detect if the hand is making a drawing gesture"""
        # Drawing gesture: Index finger extended, middle finger down
        # Check if index finger is extended (tip higher than knuckle)
        index_extended = index_tip[1] < index_mcp[1] - 40
        
        # Check if middle finger is not extended (close to knuckle)
        middle_distance = np.sqrt((middle_tip[0] - index_mcp[0])**2 + (middle_tip[1] - index_mcp[1])**2)
        middle_not_extended = middle_distance < 100
        
        return index_extended and middle_not_extended
    
    def is_eraser_gesture(self, index_tip, index_mcp, middle_tip):
        """Detect if the hand is making an eraser gesture"""
        # Eraser gesture: Both index and middle fingers extended
        index_extended = index_tip[1] < index_mcp[1] - 40
        middle_extended = middle_tip[1] < index_mcp[1] - 40
        
        return index_extended and middle_extended
    
    def draw_on_canvas(self, point, is_drawing):
        """Draw on the canvas based on finger position"""
        # If we're starting to draw, don't connect from the previous point
        if not self.drawing_points or not is_drawing:
            self.drawing_points.append(point)
            return
        
        # Add current point to drawing points
        self.drawing_points.append(point)
        
        # Draw line between last two points
        if self.mode == "draw":
            cv2.line(self.canvas, self.drawing_points[-2], self.drawing_points[-1], 
                   self.current_color, self.brush_thickness)
        elif self.mode == "erase":
            # Erase by drawing black circles (same as background)
            cv2.circle(self.canvas, point, 20, (0, 0, 0), -1)
    
    def draw_ui(self, frame):
        """Draw UI elements on the frame"""
        # Draw FPS
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time) if self.curr_time != self.prev_time else 0
        self.prev_time = self.curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw current mode
        mode_text = f"Mode: {self.mode.capitalize()}"
        cv2.putText(frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw current color indicator
        color_name = next((name for name, color in self.colors.items() if np.array_equal(color, self.current_color)), "unknown")
        color_text = f"Color: {color_name.capitalize()}"
        cv2.putText(frame, color_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw color palette
        y_pos = 150
        for i, (color_name, color_value) in enumerate(self.colors.items()):
            if color_name != "eraser":  # Skip eraser in color palette
                cv2.circle(frame, (30, y_pos), 15, color_value, -1)
                if np.array_equal(color_value, self.current_color):
                    cv2.circle(frame, (30, y_pos), 18, (255, 255, 255), 2)  # Highlight selected color
                y_pos += 40
        
        # Draw help text
        cv2.putText(frame, "Controls:", (self.width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "- Index finger: Draw", (self.width - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "- Two fingers: Erase", (self.width - 300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "- Press 'c': Clear canvas", (self.width - 300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "- Press 's': Save canvas", (self.width - 300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "- Press 'q': Quit", (self.width - 300, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
    def save_canvas(self):
        """Save the current canvas as an image"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.output_dir, f"canvas_{timestamp}.png")
        cv2.imwrite(filename, self.canvas)
        return filename
    
    def run(self):
        """Main function to run the air canvas"""
        # Configure MediaPipe hands
        hands_config = {
            'max_num_hands': 1,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.7
        }
        
        with self.mp_hands.Hands(**hands_config) as hands:
            while self.cap.isOpened():
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip the frame horizontally for a natural selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame to detect hands
                results = hands.process(rgb_frame)
                
                # Get hand landmarks if detected
                hand_landmarks = results.multi_hand_landmarks
                
                # Visualize hand landmarks on the frame
                self.detect_hand_landmarks(frame, hand_landmarks)
                
                # Update drawing based on hand position
                if hand_landmarks:
                    # Get finger positions
                    index_tip, index_mcp, middle_tip = self.get_finger_positions(hand_landmarks)
                    
                    # Determine gesture
                    if self.is_eraser_gesture(index_tip, index_mcp, middle_tip):
                        self.mode = "erase"
                        self.is_drawing = True
                        self.draw_on_canvas(index_tip, self.is_drawing)
                    elif self.is_drawing_gesture(index_tip, index_mcp, middle_tip):
                        self.mode = "draw"
                        self.is_drawing = True
                        self.draw_on_canvas(index_tip, self.is_drawing)
                    else:
                        self.is_drawing = False
                        self.drawing_points = []
                else:
                    # Reset drawing state when no hand is detected
                    self.is_drawing = False
                    self.drawing_points = []
                
                # Combine frame and canvas
                combined_frame = cv2.addWeighted(frame, 1.0, self.canvas, 0.7, 0)
                
                # Draw UI elements
                self.draw_ui(combined_frame)
                
                # Display the result
                cv2.imshow('Air Canvas', combined_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Clear canvas
                    self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    self.drawing_points = []
                elif key == ord('s'):
                    # Save canvas
                    filename = self.save_canvas()
                    print(f"Canvas saved as {filename}")
                elif key == ord('b'):
                    self.current_color = self.colors['blue']
                elif key == ord('g'):
                    self.current_color = self.colors['green']
                elif key == ord('r'):
                    self.current_color = self.colors['red']
                elif key == ord('y'):
                    self.current_color = self.colors['yellow']
                elif key == ord('p'):
                    self.current_color = self.colors['purple']
                elif key == ord('o'):
                    self.current_color = self.colors['orange']
                elif key == ord('w'):
                    self.current_color = self.colors['white']
                elif key == ord('+') or key == ord('='):
                    # Increase brush thickness
                    self.brush_thickness = min(30, self.brush_thickness + 1)
                elif key == ord('-'):
                    # Decrease brush thickness
                    self.brush_thickness = max(1, self.brush_thickness - 1)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        air_canvas = AirCanvas()
        air_canvas.run()
    except Exception as e:
        print(f"Error: {e}")