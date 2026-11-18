# MediaWave: Real-Time Hand Gesture Recognition for Media Control

Control your media player without touching the keyboard. MediaWave tracks your hand through a standard webcam, classifies the gesture with a machine learning model, and sends the matching media command to your computer in real time.

## Gestures

| Gesture | Action | Behaviour |
|---|---|---|
| Open palm | Play / Pause | Fires once; drop or change the gesture to fire again |
| Peace sign | Next track | Fires once (needs a playlist) |
| Thumbs up | Volume up | Repeats every 0.4 s while held |
| Thumbs down | Volume down | Repeats every 0.4 s while held |
| Fist | 2x playback speed | Stays at 2x until your hand leaves the screen, then returns to 1x |

## How it works

1. **OpenCV** captures webcam frames.
2. **MediaPipe Hands** detects the hand and returns 21 landmarks (x, y, z).
3. Landmarks become a **63-value feature vector**, shifted so the wrist is the origin and scaled by the wrist-to-middle-finger-base distance. This makes it independent of hand position and distance from the camera.
4. A **Random Forest** (Scikit-Learn) predicts the gesture and a confidence score.
5. A gesture counts only if confidence is at least 75% and it is stable for 5 consecutive frames.
6. **PyAutoGUI** sends the matching media key or shortcut.

## Tech stack

Python 3.11, OpenCV, MediaPipe, NumPy, Pandas, Scikit-Learn, Joblib, PyAutoGUI, PyGetWindow

## Setup (Windows)

```powershell
git clone https://github.com/vaishnavir445-gif/mediawave-gesture-control.git
cd mediawave-gesture-control
pip install -r requirements.txt
```

## Usage

The repository already includes a trained model, so you can go straight to step 4.

1. `python scripts/1_test_webcam.py`: check that hand tracking works
2. `python scripts/2_collect_data.py`: record samples for one gesture (run once per gesture, type its label)
3. `python scripts/3_train_model.py`: train and evaluate the classifier
4. `python scripts/4_live_control.py`: run live gesture control

For the 2x speed gesture, open YouTube with the video at 1x speed and keep that tab active in its browser window. Press `q` in the preview window, or `Ctrl+C` in the terminal, to quit.

## Project structure

```
mediawave-gesture-control/
├── data/        # recorded landmark dataset (CSV)
├── models/      # trained model (gesture_model.pkl)
├── scripts/     # the four pipeline scripts + remove_label.py utility
├── requirements.txt
└── README.md
```

## Results and limitations

- Dataset: about 2,800 labelled samples across 5 gestures, recorded by one user in one lighting setup.
- Evaluation: 80/20 held-out split with a confusion matrix (see `3_train_model.py` output). The gestures were cleanly separated, but because training and test data come from the same session, this does not guarantee performance for other users or lighting.
- Windows only (uses Windows media keys and window focusing).
- The 2x speed feature uses YouTube's `Shift+.` / `Shift+,` shortcuts and assumes the video starts at 1x.

## Future work

- Test across more users, lighting conditions and camera angles, and measure FPS
- Customizable gestures and key mappings
- Support for more media players and operating systems
- Dynamic (motion-based) gestures

