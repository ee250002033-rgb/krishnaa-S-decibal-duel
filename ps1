!pip install -q kaggle librosa tensorflow pandas scikit-learn tqdm

import os, glob, librosa, numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from google.colab import files

files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c the-frequency-quest -p /content/
!unzip -q /content/the-frequency-quest.zip -d /content/dataset

train_candidates = glob.glob("/content/dataset/**/[Tt]rain", recursive=True)
test_candidates = glob.glob("/content/dataset/**/[Tt]est", recursive=True)

TRAIN_DIR = sorted(train_candidates, key=lambda x: x.count('/'))[-1]
TEST_DIR  = sorted(test_candidates, key=lambda x: x.count('/'))[-1]

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

for label in os.listdir(TRAIN_DIR):
    class_dir = os.path.join(TRAIN_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    audio_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg'))]

    for file in tqdm(audio_files):
        mel = extract_mel(os.path.join(class_dir, file))
        if mel is not None:
            X.append(mel)
            y.append(label)

if len(X) == 0:
    raise ValueError("No audio files loaded")

X = np.array(X)[..., np.newaxis]

le = LabelEncoder()
y_encoded = tf.keras.utils.to_categorical(le.fit_transform(y))

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

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
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

model.compile(
    optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=90,
    batch_size=32,
    verbose=1
)

model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    verbose=1
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

model.save("my_best_audio_model.h5")

X_test_data = []
test_files = sorted(os.listdir(TEST_DIR))

for file in tqdm(test_files):
    mel = extract_mel(os.path.join(TEST_DIR, file))
    if mel is None:
        mel = np.zeros(IMG_SIZE)
    X_test_data.append(mel)

X_test_data = np.array(X_test_data)[..., np.newaxis]

pred = model.predict(X_test_data)
pred_labels = le.inverse_transform(np.argmax(pred, axis=1))

submission = pd.DataFrame({
    "ID": test_files,
    "Class": pred_labels
})

submission.to_csv("submission.csv", index=False)

files.download("submission.csv")
