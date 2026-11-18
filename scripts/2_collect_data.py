"""
Step 2: Collect labeled hand landmark data for training.

What this does:
- Asks you which gesture you're recording (e.g. "open_palm")
- Opens your webcam
- Every frame it sees your hand, it extracts 21 landmarks (x, y, z each)
  and saves them as one row in a CSV, labeled with the gesture name
- Press 'q' to stop recording for this gesture

Run this ONCE PER GESTURE. Example plan:
    python 2_collect_data.py   -> type "open_palm"   -> hold open palm, record ~300 samples
    python 2_collect_data.py   -> type "fist"         -> hold fist, record ~300 samples
    python 2_collect_data.py   -> type "peace"         -> hold peace sign
    python 2_collect_data.py   -> type "thumbs_up"
    python 2_collect_data.py   -> type "thumbs_down"

Each run APPENDS to the same CSV (data/gesture_data.csv), so running it
multiple times for the same label just adds more samples (more data = better).

TIP: While recording, slowly move your hand slightly — rotate it a little,
move closer/farther from the camera, change the angle a bit. This helps
the model generalize instead of memorizing one exact hand position.
"""

import cv2
import mediapipe as mp
import csv
import os

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "gesture_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Setup MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


def extract_landmark_row(hand_landmarks):
    """
    Convert MediaPipe's 21 landmarks into a flat list of numbers.

    Each landmark has x, y, z (all normalized 0-1 relative to the image).
    We flatten all 21 into one row: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    That's 63 numbers total per sample.

    NOTE: We are NOT normalizing relative to the wrist yet — we'll do that
    later in the training script, since it's easier to experiment with
    different normalization strategies if we keep the raw data too.
    """
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return row


def main():
    label = input("Enter the gesture label you're recording (e.g. open_palm, fist, peace, thumbs_up, thumbs_down): ").strip()
    if not label:
        print("No label entered, exiting.")
        return

    # Create CSV with header if it doesn't exist yet
    file_exists = os.path.isfile(CSV_PATH)
    csv_file = open(CSV_PATH, mode="a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        header = ["label"]
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print(f"\nRecording gesture: '{label}'")
    print("Hold the gesture in front of the camera. Move it slightly (angle/distance) for variety.")
    print("Press 'q' to stop recording.\n")

    sample_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand skeleton so you can see what's being tracked
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Save this frame's landmarks as one training sample
                row = extract_landmark_row(hand_landmarks)
                writer.writerow([label] + row)
                sample_count += 1

        # Show live sample count on screen so you know how much data you've collected
        cv2.putText(
            frame,
            f"Label: {label} | Samples: {sample_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' to stop",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Step 2 - Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    print(f"\nDone! Recorded {sample_count} samples for label '{label}'.")
    print(f"Saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
