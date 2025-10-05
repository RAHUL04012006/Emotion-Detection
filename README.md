# Emotion Detection Using CNN (FER2013)

This project implements a Convolutional Neural Network (CNN) for facial emotion detection using the FER2013 dataset, with real-time webcam inference and music suggestions based on detected emotion.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Real-time Emotion Detection & Music Suggestion](#2-real-time-emotion-detection--music-suggestion)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

The Emotion Detection project uses deep learning to classify facial emotions from images and webcam streams. It leverages a CNN built with TensorFlow and Keras to recognize **7 emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. After detecting an emotion in real time, the app suggests a song to match the user's mood.

---

## Features

- **CNN Model** for robust emotion classification
- **Data Augmentation** for improved generalization
- **Real-time webcam detection** using OpenCV
- **Music suggestion** mapped to detected emotion

---

## Dataset

- **FER2013**: [Download from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
  - 35,887 grayscale, 48x48 pixel face images in 7 emotion categories.
  - After download, extract so that you have `train/` and `test/` directories as required by Keras' `flow_from_directory`.

---

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/RAHUL04012006/Emotion-Detection.git
   cd Emotion-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow keras opencv-python numpy
   ```

3. **Download and Extract FER2013:**
   - Place the `train/` and `test/` folders as described above in your working directory.

---

## Usage

### 1. Training the Model

Run the notebook to train the CNN on FER2013:

```bash
jupyter notebook dl.ipynb
```

- The notebook builds and trains a CNN on the FER2013 data.
- After training, the model is saved as `emotion_detection_model.h5`.

### 2. Real-time Emotion Detection & Music Suggestion

- The notebook (or code block) also includes a section for real-time webcam emotion detection.
- For live demo, ensure your webcam is connected and run the section "STEP 2: Real-time Detection + Music Suggestion".
- For each detected face, the system:
  - Predicts emotion
  - Suggests a song matching the current mood (shown as overlay text)

**Stop the webcam** by pressing `q`.

---

## Project Structure

```
.
├── dl.ipynb                # Main notebook (training + inference)
├── emotion_detection_model.h5  # Saved model (after training)
├── train/                  # Training images (FER2013)
├── test/                   # Test images (FER2013)
```

---

## Requirements

- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- NumPy

Install all Python dependencies with:
```bash
pip install tensorflow keras opencv-python numpy
```

---

## Acknowledgements

- **FER2013 Dataset**: [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Keras & TensorFlow** for model building
- **OpenCV** for real-time face detection

---

## License

This project is for educational purposes. License information to be added.

---
