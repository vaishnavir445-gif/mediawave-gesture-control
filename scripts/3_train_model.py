"""
Step 3: Train a gesture classifier on the collected landmark data.

What this does:
- Loads data/gesture_data.csv
- Normalizes each hand's landmarks relative to the wrist (landmark 0),
  so the model learns hand SHAPE, not hand POSITION in the frame
- Splits data into training (80%) and testing (20%) sets
- Trains a Random Forest classifier
- Prints accuracy + a confusion matrix so we can see which gestures
  (if any) get confused with each other
- Saves the trained model to models/gesture_model.pkl

Run this after you've collected all 5 gestures:
    python 3_train_model.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CSV_PATH = os.path.join(DATA_DIR, "gesture_data.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")
os.makedirs(MODELS_DIR, exist_ok=True)


def normalize_landmarks(row):
    """
    Take one row of 63 raw landmark values (21 points x,y,z) and make it
    POSITION-INVARIANT and SCALE-INVARIANT.

    Why this matters: right now, x0,y0,z0 (wrist) could be anywhere in the
    frame depending on where you held your hand. If we train on raw
    coordinates, the model might partly learn "gestures recorded on the
    left side of my webcam" instead of "gesture shape" - not what we want.

    Fix:
    1. Subtract the wrist position from every point (so wrist becomes 0,0,0
       and everything else is relative to it)
    2. Divide by the distance from wrist to middle-finger-base (landmark 9)
       to normalize for hand size / distance from camera
    """
    row = np.array(row, dtype=np.float64).reshape(21, 3)  # reshape flat list into 21 (x,y,z) points

    wrist = row[0].copy()
    row -= wrist  # step 1: make wrist the origin

    # step 2: scale normalize using wrist -> middle finger base distance
    scale_ref = np.linalg.norm(row[9])  # landmark 9 = base of middle finger
    if scale_ref < 1e-6:  # avoid divide-by-zero on bad data
        scale_ref = 1e-6
    row /= scale_ref

    return row.flatten()  # back to a flat list of 63 numbers


def main():
    if not os.path.isfile(CSV_PATH):
        print(f"No data found at {CSV_PATH}. Run 2_collect_data.py first for each gesture.")
        return

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} raw rows")

    # --- Clean up bad rows ---
    # If the CSV ever got a stray/duplicate header row mixed into the data
    # (e.g. from re-saving in Excel, or appending twice), the landmark
    # columns will contain non-numeric junk like the word "label" or a
    # gesture name instead of a number. We detect and drop those rows here.
    landmark_cols = [c for c in df.columns if c != 'label']

    before = len(df)
    for col in landmark_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # turns bad values into NaN
    df = df.dropna(subset=landmark_cols)  # drop any row that had a bad value
    after = len(df)

    if before != after:
        print(f"Dropped {before - after} corrupted/bad rows (non-numeric landmark values).")

    print(f"Clean dataset: {len(df)} samples across {df['label'].nunique()} labels: {sorted(df['label'].unique())}")
    print(df['label'].value_counts())

    # Separate labels from landmark data
    labels = df['label'].values
    raw_landmarks = df[landmark_cols].values

    print("\nNormalizing landmarks (position + scale invariant)...")
    normalized = np.array([normalize_landmarks(row) for row in raw_landmarks])

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        normalized, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Test Accuracy: {acc*100:.2f}% ===\n")

    print("Classification report (precision/recall per gesture):")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix (rows = actual, columns = predicted):")
    labels_sorted = sorted(df['label'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    print(cm_df)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
