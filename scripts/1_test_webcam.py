"""
Step 1: Test webcam + MediaPipe hand tracking.

What this does:
- Opens your default webcam
- Detects your hand in each frame
- Draws 21 landmark points + connecting lines on your hand (skeleton view)
- Press 'q' to quit

If you see dots tracking your hand in real time, you're good to move to Step 2.
"""

import cv2
import mediapipe as mp

# --- Setup MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,   # False = video stream mode (uses tracking between frames, faster)
    max_num_hands=1,           # only track 1 hand for now, keep it simple
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# --- Open webcam ---
cap = cv2.VideoCapture(0)  # 0 = default webcam. Try 1 if this doesn't work.

if not cap.isOpened():
    print("ERROR: Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1).")
    exit()

print("Webcam started. Press 'q' in the video window to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    # Flip horizontally so it feels like a mirror (more natural for the user)
    frame = cv2.flip(frame, 1)

    # MediaPipe expects RGB, OpenCV gives us BGR — convert
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run hand detection
    results = hands.process(rgb_frame)

    # If a hand was found, draw the landmarks on the original (BGR) frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    cv2.imshow("Step 1 - Hand Tracking Test", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
