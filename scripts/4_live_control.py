"""
Step 4: Live gesture recognition -> real media control.

What this does:
- Opens your webcam
- Every frame: detects hand landmarks -> normalizes them (same way as training)
  -> feeds them to your trained model -> gets a predicted gesture + confidence
- Uses a small "stability buffer" so a gesture only counts once it's been
  seen consistently for a few frames in a row (avoids jittery false triggers)
- Maps each stable gesture to a real keyboard action:
    open_palm    -> Play/Pause (EDGE-triggered: fires once, then locks until
                    you change gesture - won't repeat while held)
    peace        -> Next track (COOLDOWN-triggered: repeats while held, throttled)
    thumbs_up    -> Volume Up (COOLDOWN-triggered: repeats while held, throttled)
    thumbs_down  -> Volume Down (COOLDOWN-triggered: repeats while held, throttled)
    fist         -> 2x speed; stays on until your hand leaves the screen, then back to 1x

IMPORTANT: The exact keys for "next track" / "speed up" etc. depend on
which media player/app you're controlling (YouTube, Spotify, VLC all
differ slightly). Check the KEY MAPPING section below and adjust if needed
for your target app - comments explain what to change.

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
import pyautogui

try:
    import pygetwindow as gw   # used to bring the YouTube window to the front
except ImportError:
    gw = None

# --- Paths ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gesture_model.pkl")

# --- Load trained model ---
if not os.path.isfile(MODEL_PATH):
    print(f"No trained model found at {MODEL_PATH}. Run 3_train_model.py first.")
    exit()

model = joblib.load(MODEL_PATH)
print("Model loaded.")

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

# ============== KEY MAPPING (edit these if your app uses different shortcuts) ==============
# EDGE-TRIGGERED: fires ONCE when the gesture appears, then "locks" until you
# change gesture / lower your hand. Won't fire again just because you keep
# holding the same pose. Good for toggle-style actions like play/pause.
EDGE_TRIGGER_ACTIONS = {
    "open_palm": "playpause",    # Play / Pause - system media key
    "peace": "nexttrack",        # Next track (media key - works system-wide on Windows)
}

# COOLDOWN-TRIGGERED: fires repeatedly while held, once every TRIGGER_COOLDOWN
# seconds. Good for actions you'd want to repeat by holding, like volume.
COOLDOWN_TRIGGER_ACTIONS = {
    "thumbs_up": "volumeup",     # Volume up (system volume key)
    "thumbs_down": "volumedown", # Volume down (system volume key)
}
TRIGGER_COOLDOWN = 0.4  # seconds between repeats while a cooldown gesture is held


# "Hold" gesture: fist -> speed up to 2x while held, back to 1x on release.
# YouTube's player has NO single "hold key = 2x" shortcut, but it does support
# Shift+. (increase speed one step) and Shift+, (decrease speed one step).
# YouTube's speed steps from 1.0x are: 1.25, 1.5, 1.75, 2.0 - so pressing
# Shift+. four times in a row takes you from 1.0x -> 2.0x, and Shift+,
# four times brings it back down to 1.0x.
# CAVEAT: this assumes the video was at normal 1x speed before you started -
# make sure playback speed is reset to 1x before testing/demoing this.
HOLD_GESTURE = "fist"
SPEED_STEPS = 4              # number of Shift+. presses to go from 1x -> 2x
HAND_GONE_FRAMES = 20        # hand must be OUT of the screen this many frames in a row (~0.7s) to drop back to 1x
WINDOW_NAME = "Step 4 - Live Gesture Control"   # our webcam preview window
WINDOW_KEYWORD = "YouTube"   # browser window whose title contains this gets focused before speed keys
KEY_DELAY = 0.1              # seconds between speed keypresses (helps YouTube register each one)
SPEED_UP_KEYS = ("shift", ".")
SPEED_DOWN_KEYS = ("shift", ",")
# =============================================================================================

# Confidence + stability settings
CONFIDENCE_THRESHOLD = 0.75   # model must be at least this sure
STABILITY_FRAMES = 5          # gesture must appear this many consecutive frames to count


def focus_youtube():
    """Bring the browser window showing YouTube to the front so Shift+. / Shift+, reach it.
    (Media keys work from any window, but YouTube's speed shortcuts need the page focused.)
    The YouTube tab must be the ACTIVE tab of its browser window."""
    if gw is None:
        return
    try:
        wins = [w for w in gw.getWindowsWithTitle(WINDOW_KEYWORD) if w.title]
        if wins:
            w = wins[0]
            if w.isMinimized:
                w.restore()
            w.activate()
    except Exception:
        pass  # pygetwindow sometimes raises even when activation worked
    time.sleep(0.25)


def focus_preview():
    """Give keyboard focus back to our webcam window (so pressing 'q' still works)."""
    if gw is None:
        return
    try:
        wins = gw.getWindowsWithTitle(WINDOW_NAME)
        if wins:
            wins[0].activate()
    except Exception:
        pass


def change_speed(keys):
    focus_youtube()
    for _ in range(SPEED_STEPS):
        pyautogui.hotkey(*keys)
        time.sleep(KEY_DELAY)
    focus_preview()


def normalize_landmarks(hand_landmarks):
    """Same normalization used during training - MUST match exactly."""
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    row = np.array(row, dtype=np.float64).reshape(21, 3)

    wrist = row[0].copy()
    row -= wrist

    scale_ref = np.linalg.norm(row[9])
    if scale_ref < 1e-6:
        scale_ref = 1e-6
    row /= scale_ref

    return row.flatten().reshape(1, -1)  # reshape for sklearn's .predict() which expects 2D input


def main():
    # Small preview window pinned to the top-left corner, always on top,
    # so it stays visible even when the browser is brought to the front.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 480, 360)
    screen_w, screen_h = pyautogui.size()
    cv2.moveWindow(WINDOW_NAME, 20, 20)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass  # older OpenCV builds don't support always-on-top

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("\nLive gesture control started. Press 'q' to quit.\n")

    recent_predictions = []   # small rolling buffer for stability check
    last_fired_gesture = None # which EDGE-triggered gesture is currently "claimed" -
                               # stays claimed until gesture changes, so it won't re-fire
    last_trigger_time = {}    # per-gesture cooldown timestamps for COOLDOWN-triggered actions
    hold_active = False       # tracks whether we're currently in "2x speed" mode
    hand_gone_count = 0       # consecutive frames with no hand on screen (used to end 2x mode)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_gesture = None
        current_confidence = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # only using first detected hand

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            features = normalize_landmarks(hand_landmarks)
            probs = model.predict_proba(features)[0]
            best_idx = np.argmax(probs)
            current_gesture = model.classes_[best_idx]
            current_confidence = probs[best_idx]

            if current_confidence < CONFIDENCE_THRESHOLD:
                current_gesture = None  # not confident enough, treat as "no gesture"

        # --- Update stability buffer ---
        recent_predictions.append(current_gesture)
        if len(recent_predictions) > STABILITY_FRAMES:
            recent_predictions.pop(0)

        # Gesture is "stable" only if the last STABILITY_FRAMES were all the same non-None gesture
        stable_gesture = None
        if len(recent_predictions) == STABILITY_FRAMES and len(set(recent_predictions)) == 1:
            stable_gesture = recent_predictions[0]

        # --- Handle 2x SPEED mode ---
        # Show a stable fist  -> speed goes to 2x and STAYS at 2x (you can use other
        #                        gestures like volume while it's on).
        # Take your hand fully out of the screen for ~0.7s -> speed returns to 1x.
        action_text = ""
        if results.multi_hand_landmarks:
            hand_gone_count = 0
        else:
            hand_gone_count += 1

        if stable_gesture == HOLD_GESTURE and not hold_active:
            change_speed(SPEED_UP_KEYS)
            hold_active = True
            action_text = "2x speed ON (fist)"
            print(action_text)
        elif hold_active and hand_gone_count >= HAND_GONE_FRAMES:
            change_speed(SPEED_DOWN_KEYS)
            hold_active = False
            recent_predictions.clear()
            action_text = "2x speed OFF (hand left screen)"
            print(action_text)

        # --- Handle EDGE-triggered gestures (fire once per gesture "session") ---
        # Fires only when the stable gesture CHANGES to a new edge-trigger gesture.
        # To fire it again, you need to drop the gesture (change pose / lower hand) first.
        if stable_gesture in EDGE_TRIGGER_ACTIONS and stable_gesture != last_fired_gesture:
            key_to_press = EDGE_TRIGGER_ACTIONS[stable_gesture]
            pyautogui.press(key_to_press)
            action_text = f"Triggered: {stable_gesture} -> {key_to_press}"
            print(action_text)

        # Reset the "claim" once the stable gesture is no longer that edge gesture,
        # so the same gesture CAN fire again next time it's made
        if stable_gesture in EDGE_TRIGGER_ACTIONS:
            last_fired_gesture = stable_gesture
        elif stable_gesture is None:
            last_fired_gesture = None

        # --- Handle COOLDOWN-triggered gestures (repeat while held, throttled) ---
        if stable_gesture in COOLDOWN_TRIGGER_ACTIONS:
            now = time.time()
            last_time = last_trigger_time.get(stable_gesture, 0)
            if now - last_time > TRIGGER_COOLDOWN:
                key_to_press = COOLDOWN_TRIGGER_ACTIONS[stable_gesture]
                pyautogui.press(key_to_press)
                last_trigger_time[stable_gesture] = now
                action_text = f"Triggered: {stable_gesture} -> {key_to_press}"
                print(action_text)

        # --- On-screen debug info ---
        display_text = f"Gesture: {current_gesture or '-'} ({current_confidence*100:.0f}%)"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if hold_active:
            cv2.putText(frame, "HOLD ACTIVE (fist)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if action_text:
            cv2.putText(frame, action_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Safety: make sure playback isn't left stuck at 2x speed if you quit mid-fist
    if hold_active:
        change_speed(SPEED_DOWN_KEYS)

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
