# 🎧 Decibel Duel — Sound Classification Project

This project an attempt at building a machine-learning model that can listen to short audio clips and guess what kind of sound it is.
I used the Kaggle competition page directly to downloaded the dataset, converted the audio into pictures called mel-spectrograms, and then trained a CNN (a type of neural network) to classify the sounds.

## 1. Installing the Libraries

Before doing anything, we install all the Python packages needed:

`kaggle` to download the dataset

`librosa` to read audio

`tensorflow` for the neural network

`pandas`, `numpy`, `sklearn`, and `tqdm` for data handling*

## 2. Setting Up Kaggle

To download the data automatically, the code asks you to upload your `kaggle.json` file.
This file basically lets your notebook talk to Kaggle.
After uploading, we put it in the right folder so it works properly.

## 3. Downloading the Dataset

We download the competition zip file using your Kaggle account and unzip it:

Now we have folders called train and test.

The code also auto-detects the correct train/test directories even if they are nested inside other folders.

## 4. Turning Audio into Images

Neural networks like images more than raw audio, so we convert each sound file into a mel-spectrogram.
You can think of it as a colorful picture showing how strong different frequencies are at every moment.

This is done using a helper function:

def extract_mel(file):
    y, sr = librosa.load(file, sr=22050, duration=3)
    mel = librosa.feature.melspectrogram(...)
    mel_db = librosa.power_to_db(mel)
    mel_db = np.resize(mel_db, (128,128))
    return mel_db


Each mel-spectrogram becomes a 128×128 pixel image.

## 5. Loading and Preparing the Data

The code loops through each sound class (like dog_bark, siren, drilling, etc.) in the training folder.
For every audio file:

Load the audio

Convert it to mel-spectrogram

Add it to a list

Then we convert the labels into numbers so the model can understand them.

## 6. Train-Validation Split

We split the data into:

Training data (80%)

Validation data (20%)

This helps us track how well the model is learning.

## 7. Building the CNN Model

The model is made of multiple layers:

Convolution layers (to understand the spectrogram images)

Batch normalization (to help stabilize training)

MaxPooling (to shrink the image and keep important info)

Dense layers at the end to make the final prediction

Dropout (to reduce overfitting)


## 8. Training the Model

Training happens in two steps:

### Step 1: Warm-up with Adam optimizer

The Adam optimizer helps the model learn quickly in the beginning.

We train for 10 epochs.

### Step 2: Fine-tuning with SGD optimizer

SGD (with momentum) makes the learning more stable and helps fix overfitting.

We also use:

Early stopping: stops training if things stop improving

Learning rate scheduler: automatically slows down learning if needed

This part trains up to 60 epochs.

## 9. Checking Accuracy

After training, the code prints the best validation accuracy the model reached.
In my run, it got around 93% accuracy

## 10. Predicting on Test Set

For each audio file in the test folder:

Convert it into a mel-spectrogram

Pass it through the model

Get the predicted class

We store everything in a CSV file:

submission.csv


## 11. Saving Features for Future Runs

Since feature extraction takes time, I saved the processed data:

np.save("X_features.npy", X)
np.save("y_labels.npy", y_encoded)


Next time, I just load these and skip all the heavy extraction steps.
so next time I only need to run the cell that has my model code and not load the API or preprocess the data again
