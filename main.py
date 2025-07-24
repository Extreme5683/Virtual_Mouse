import time
import cv2
import mediapipe as mp
import pyautogui

# Get screen size
screen_width, screen_height = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Smoothing variables
prev_x, prev_y = 0, 0
smooth_factor = 1  # Set to 1 for responsiveness

# Click control
last_click_time = 0
click_delay = 0.3

# Pause control
paused = False
last_toggle_time = 0
toggle_delay = 0.8  # 800ms

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    height, width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Screen coords
            screen_x = int(index_tip.x * screen_width)
            screen_y = int(index_tip.y * screen_height)

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smooth_factor
            curr_y = prev_y + (screen_y - prev_y) / smooth_factor

            curr_x = max(0, min(screen_width - 1, int(curr_x)))
            curr_y = max(0, min(screen_height - 1, int(curr_y)))

            # Click detection (pinch)
            index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)
            thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            distance = ((index_x - thumb_x)**2 + (index_y - thumb_y)**2)**0.5
            current_time = time.time()

            if distance < 30 and current_time - last_click_time > click_delay:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, 'Click!', (index_x + 30, index_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Pause/resume detection (fist)
            finger_tips = [8, 12, 16, 20]
            finger_bases = [6, 10, 14, 18]
            fist = True
            for tip, base in zip(finger_tips, finger_bases):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                    fist = False
                    break

            if fist and current_time - last_toggle_time > toggle_delay:
                paused = not paused
                last_toggle_time = current_time
                print("Mouse control PAUSED" if paused else "Mouse control RESUMED")

            # Draw landmarks and fingertips
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (index_x, index_y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 0), -1)

            # Move mouse if not paused
            if not paused:
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

    # Show pause label
    if paused:
        cv2.putText(frame, 'PAUSED', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Display webcam frame
    cv2.imshow("Virtual Mouse", frame)

    # Exit on Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
