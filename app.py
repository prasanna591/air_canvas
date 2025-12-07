# air_canvas_fixed.py
import cv2
import numpy as np
import mediapipe as mp
import time
import os
from collections import deque

class AirCanvas:
    def __init__(self, cam_index=0, width=1280, height=720):
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Video capture
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise IOError("Cannot access webcam. Make sure a camera is connected and not used by another app.")

        # Preferred resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Try to read one frame to get actual dimensions
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # fallback to requested values
            self.width = width
            self.height = height
            # initialize a blank frame to avoid None errors
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            # flip later in run() for natural selfie view, but we only need shape
            self.height, self.width = frame.shape[:2]

        # Canvas (same size as frame) - holds persistent drawing
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Drawing state
        self.drawing_points = deque(maxlen=1024)  # keep last points for smoothing if needed
        self.prev_point = None
        self.current_color = (0, 0, 255)  # BGR: red
        self.brush_thickness = 6
        self.eraser_radius = 40
        self.mode = "draw"  # "draw" or "erase"
        self.is_drawing = False

        # Timing for FPS
        self.prev_time = time.time()

        # Color palette (BGR)
        self.colors = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'orange': (0, 165, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
        }
        # Precompute palette positions (x,y,radius)
        self.palette_items = []
        start_y = 150
        x = 40
        r = 18
        for i, (name, val) in enumerate(self.colors.items()):
            self.palette_items.append((name, val, (x, start_y + i * 45), r))

        # Output dir
        self.output_dir = "canvas_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_hand_landmarks(self, image, hand_landmarks):
        if not hand_landmarks:
            return
        for hand_lms in hand_landmarks:
            # drawing utilities
            self.mp_drawing.draw_landmarks(
                image,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

    def get_finger_positions(self, hand_landmarks):
        """Return pixel coords of index_tip, index_mcp, middle_tip for first detected hand"""
        if not hand_landmarks:
            return None, None, None
        hand_lms = hand_landmarks[0]
        # safe access in case some landmarks absent
        try:
            ix = int(hand_lms.landmark[8].x * self.width)
            iy = int(hand_lms.landmark[8].y * self.height)
            index_tip = (ix, iy)

            imx = int(hand_lms.landmark[5].x * self.width)
            imy = int(hand_lms.landmark[5].y * self.height)
            index_mcp = (imx, imy)

            mx = int(hand_lms.landmark[12].x * self.width)
            my = int(hand_lms.landmark[12].y * self.height)
            middle_tip = (mx, my)
        except Exception:
            return None, None, None

        return index_tip, index_mcp, middle_tip

    def is_index_extended(self, index_tip, index_mcp):
        # index extended if tip is sufficiently above knuckle (smaller y since origin top-left)
        if index_tip is None or index_mcp is None:
            return False
        return index_tip[1] < index_mcp[1] - 30

    def is_middle_extended(self, middle_tip, index_mcp):
        if middle_tip is None or index_mcp is None:
            return False
        return middle_tip[1] < index_mcp[1] - 30

    def is_drawing_gesture(self, index_tip, index_mcp, middle_tip):
        # index extended, middle not extended -> draw
        return self.is_index_extended(index_tip, index_mcp) and not self.is_middle_extended(middle_tip, index_mcp)

    def is_eraser_gesture(self, index_tip, index_mcp, middle_tip):
        # both index and middle extended -> erase
        return self.is_index_extended(index_tip, index_mcp) and self.is_middle_extended(middle_tip, index_mcp)

    def touch_palette(self, point):
        """If point is near a palette circle, select that color and return True"""
        if point is None:
            return False
        px, py = point
        for name, val, (cx, cy), r in self.palette_items:
            dist = np.hypot(px - cx, py - cy)
            if dist <= r + 10:  # tolerance
                self.current_color = val
                self.mode = "draw"
                # small visual feedback: draw a white ring on canvas at palette (done when rendering)
                return True
        return False

    def draw_on_canvas(self, point):
        if point is None:
            self.prev_point = None
            return

        if self.mode == "draw":
            if self.prev_point is None:
                # start point
                self.prev_point = point
                # small dot
                cv2.circle(self.canvas, point, max(1, self.brush_thickness // 2), self.current_color, -1)
                return

            # only draw if moved enough to avoid dense overlapping
            if np.hypot(point[0] - self.prev_point[0], point[1] - self.prev_point[1]) >= 2:
                cv2.line(self.canvas, self.prev_point, point, self.current_color, self.brush_thickness)
                self.prev_point = point

        elif self.mode == "erase":
            # erase by setting canvas pixels to zero in the circle region
            cv2.circle(self.canvas, point, self.eraser_radius, (0, 0, 0), -1)
            # reset prev_point to avoid accidental connecting when switching back
            self.prev_point = None

    def overlay_canvas_on_frame(self, frame):
        """Overlay only the non-black pixels from canvas onto frame using alpha blending"""
        # create mask where canvas has drawing (non-black)
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        mask = gray > 10  # threshold: anything >10 considered drawing
        mask_3 = np.dstack([mask] * 3)  # shape (h,w,3) boolean

        # convert to float for blending
        overlay = frame.copy().astype(np.float32)
        canvas_f = self.canvas.astype(np.float32)

        alpha = 0.85  # canvas visibility when drawing present
        # blend only where mask True
        overlay[mask_3] = cv2.addWeighted(frame, 1.0 - alpha, self.canvas, alpha, 0)[mask_3]

        return overlay.astype(np.uint8)

    def draw_ui(self, frame):
        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time) if curr_time != self.prev_time else 0.0
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        mode_text = f"Mode: {self.mode.capitalize()}"
        cv2.putText(frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # show selected color name
        color_name = next((n for n, v in self.colors.items() if v == self.current_color), "custom")
        cv2.putText(frame, f"Color: {color_name}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # palette
        for name, val, (cx, cy), r in self.palette_items:
            cv2.circle(frame, (cx, cy), r, val, -1)
            # highlight selected color
            if self.current_color == val:
                cv2.circle(frame, (cx, cy), r + 4, (255, 255, 255), 2)
            cv2.putText(frame, name[0].upper(), (cx + 25, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # help text
        cv2.putText(frame, "Controls: 'c' clear, 's' save, 'q' quit, +/- brush", (self.width - 650, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "Index finger = draw | Index+Middle = erase | Tap palette to change color", (self.width - 650, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    def save_canvas(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.output_dir, f"canvas_{timestamp}.png")
        cv2.imwrite(filename, self.canvas)
        return filename

    def run(self):
        hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Warning: failed to read frame from camera")
                    break

                # mirror for selfie view
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                hand_landmarks = results.multi_hand_landmarks

                # draw landmarks for UX
                self.detect_hand_landmarks(frame, hand_landmarks)

                index_tip, index_mcp, middle_tip = self.get_finger_positions(hand_landmarks)

                # Gesture resolution order: palette tap -> erase -> draw
                gesture_handled = False

                # If index is extended and not middle, but tip near palette -> change color
                if index_tip and (self.is_index_extended(index_tip, index_mcp) and not self.is_middle_extended(middle_tip, index_mcp)):
                    if self.touch_palette(index_tip):
                        # feedback: small white dot on canvas at palette pos (visual only)
                        gesture_handled = True
                        self.is_drawing = False
                        self.prev_point = None

                if not gesture_handled and index_tip is not None:
                    if self.is_eraser_gesture(index_tip, index_mcp, middle_tip):
                        self.mode = "erase"
                        self.is_drawing = True
                        self.draw_on_canvas(index_tip)
                    elif self.is_drawing_gesture(index_tip, index_mcp, middle_tip):
                        self.mode = "draw"
                        self.is_drawing = True
                        self.draw_on_canvas(index_tip)
                    else:
                        # not in drawing posture -> stop drawing & reset prev_point to avoid long lines
                        self.is_drawing = False
                        self.prev_point = None
                else:
                    # no hand detected
                    self.is_drawing = False
                    self.prev_point = None

                # Overlay canvas onto frame (only where canvas has drawing)
                combined = self.overlay_canvas_on_frame(frame)

                # Draw UI on top
                self.draw_ui(combined)

                cv2.imshow("Air Canvas", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    self.prev_point = None
                elif key == ord('s'):
                    fname = self.save_canvas()
                    print(f"Saved: {fname}")
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
                    self.brush_thickness = min(60, self.brush_thickness + 1)
                elif key == ord('-'):
                    self.brush_thickness = max(1, self.brush_thickness - 1)
                elif key == ord('e'):
                    # quick toggle eraser
                    self.mode = "erase"
                elif key == ord('d'):
                    self.mode = "draw"

        finally:
            hands.close()
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        ac = AirCanvas()
        ac.run()
    except Exception as e:
        print("Error:", e)
