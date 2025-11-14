==============================================================
# 🎧 DECIBEL DUEL 2025 - FIXED HIGH ACCURACY MODEL (Nested train/train)
# ==============================================================

!pip install -q kaggle librosa tensorflow pandas scikit-learn tqdm

import os, glob, librosa, numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras import layers, models
from google.colab import files

# --------------------------------------------------------------
# STEP 1️⃣: Upload Kaggle API key
# --------------------------------------------------------------
print("📂 Upload your kaggle.json file (Kaggle → Account → Create API Token)")
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
print("✅ Kaggle API configured!")

# --------------------------------------------------------------
# STEP 2️⃣: Download & Extract Dataset
# --------------------------------------------------------------
!kaggle competitions download -c the-frequency-quest -p /content/
!unzip -q /content/the-frequency-quest.zip -d /content/dataset
print("✅ Dataset downloaded and extracted!")

# --------------------------------------------------------------
# STEP 3️⃣: Auto-detect nested train/train & test/test
# --------------------------------------------------------------
train_candidates = glob.glob("/content/dataset/**/[Tt]rain", recursive=True)
test_candidates = glob.glob("/content/dataset/**/[Tt]est", recursive=True)

# Pick the deepest folder (handles train/train)
TRAIN_DIR = sorted(train_candidates, key=lambda x: x.count('/'))[-1]
TEST_DIR  = sorted(test_candidates, key=lambda x: x.count('/'))[-1]

print("✅ Using Train folder:", TRAIN_DIR)
print("✅ Using Test folder:", TEST_DIR)

# --------------------------------------------------------------
# STEP 4️⃣: Audio → Mel-spectrogram
# --------------------------------------------------------------
SR = 22050
DURATION = 3
N_MELS = 128
IMG_SIZE = (128, 128)

def extract_mel(file):
    try:
        y, sr = librosa.load(file, sr=SR, duration=DURATION)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.resize(mel_db, IMG_SIZE)
        return mel_db
    except Exception:
        return None

X, y = [], []

print("\n🎵 Extracting features from training data...")
for label in os.listdir(TRAIN_DIR):
    class_dir = os.path.join(TRAIN_DIR, label)
    if not os.path.isdir(class_dir): continue

    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg'))]
    print(f"📁 {label}: {len(files)} files")

    for file in tqdm(files, desc=f"Processing {label}"):
        mel = extract_mel(os.path.join(class_dir, file))
        if mel is not None:
            X.append(mel)
            y.append(label)

if len(X) == 0:
    raise ValueError("❌ No audio files were successfully loaded. Please check dataset path or file format!")

X = np.array(X)[..., np.newaxis]
le = LabelEncoder()
y_encoded = tf.keras.utils.to_categorical(le.fit_transform(y))

print(f"✅ Loaded {len(X)} samples across {len(le.classes_)} classes: {list(le.classes_)}")



X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'), # Reduced Dense layer size
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(y_encoded.shape[1], activation='softmax')
])


# --- Stage 1: Warm-up training (10 epochs) ---
print("🔥 Stage 1: Warm-up training (10 epochs)")
model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=10, batch_size=32, verbose=1)

# --- Stage 2: Medium fine-tuning (90 epochs, lower LR) ---
print("\n🎯 Stage 2: Medium fine-tuning (90 epochs)")
model.compile(optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-5), # Using AdamW as requested
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=90, batch_size=32, verbose=1)

# --- Stage 3: Deep fine-tuning (25 epochs, very low LR) ---
print("\n🏁 Stage 3: Deep fine-tuning (25 epochs)")
model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5), # Using AdamW as requested
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=25, batch_size=32, verbose=1)


# --- Final Evaluation ---
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Final Fine-Tuned Accuracy: {val_acc*100:.2f}%")

# Save the model
model.save("my_best_audio_model.h5")
print("\n💾 Model saved successfully as my_best_audio_model.h5")

# --------------------------------------------------------------
# STEP 6️⃣: Predict Test Set
# --------------------------------------------------------------
print("\n🎧 Processing test data...")
X_test_data, test_files = [], sorted(os.listdir(TEST_DIR))

for file in tqdm(test_files):
    mel = extract_mel(os.path.join(TEST_DIR, file))
    if mel is None:
        mel = np.zeros(IMG_SIZE)
    X_test_data.append(mel)

X_test_data = np.array(X_test_data)[..., np.newaxis]
pred = model.predict(X_test_data)
# Assuming le (LabelEncoder) is still available from feature extraction
# If le was not saved, you might need to reload it here or handle it differently
# Assuming le is available in the current session or re-fitted
pred_labels = le.inverse_transform(np.argmax(pred, axis=1))

submission = pd.DataFrame({"ID": test_files, "Class": pred_labels})
submission.to_csv("submission.csv", index=False)
print("\n💾 submission.csv created successfully!")

from google.colab import files
files.download("submission.csv")

print("\n🏁 DONE! Upload 'submission.csv' to Kaggle.")
