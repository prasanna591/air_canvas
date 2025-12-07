import cv2
import mediapipe as mp
import pyautogui
import math
import time

# =========================
# CONFIGURATION
# =========================
CURSOR_SMOOTHING = 0.25        # Lower = more responsive, higher = smoother
PINCH_THRESHOLD = 0.05         # Distance for a pinch gesture
RIGHT_CLICK_THRESHOLD = 0.05   # Distance for middle–thumb gesture
CLICK_DEBOUNCE_TIME = 0.25     # Seconds between allowed clicks
SCROLL_SENSITIVITY = 1200      # Adjust scroll power

# =========================
# UTILITIES
# =========================

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def lerp(a, b, t):
    """Linear smoothing."""
    return a + (b - a) * t

# =========================
# INITIALIZATION
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

prev_x, prev_y = screen_w // 2, screen_h // 2
dragging = False
last_click_time = 0

# =========================
# MAIN LOOP
# =========================
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            index_tip = hand.landmark[8]
            thumb_tip = hand.landmark[4]
            middle_tip = hand.landmark[12]

            # Convert to screen coordinates
            target_x = int(index_tip.x * screen_w)
            target_y = int(index_tip.y * screen_h)

            # Smooth motion
            cur_x = lerp(prev_x, target_x, CURSOR_SMOOTHING)
            cur_y = lerp(prev_y, target_y, CURSOR_SMOOTHING)
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            pinch_dist = distance(index_tip, thumb_tip)
            right_dist = distance(middle_tip, thumb_tip)

            now = time.time()

            # ================
            # LEFT CLICK / DRAG
            # ================
            if pinch_dist < PINCH_THRESHOLD:
                if not dragging:
                    if now - last_click_time > CLICK_DEBOUNCE_TIME:
                        pyautogui.mouseDown()
                        dragging = True
                        last_click_time = now
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # ================
            # RIGHT CLICK
            # ================
            if right_dist < RIGHT_CLICK_THRESHOLD:
                if now - last_click_time > CLICK_DEBOUNCE_TIME:
                    pyautogui.rightClick()
                    last_click_time = now

            # ================
            # SCROLL WHILE PINCHING
            # ================
            if dragging:
                scroll_amount = int((thumb_tip.y - index_tip.y) * SCROLL_SENSITIVITY)
                pyautogui.scroll(scroll_amount)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
