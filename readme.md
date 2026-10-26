# Emotion Recognition from Speech - CodeAlpha

## Project Overview
This repository contains the complete implementation of Task 2: Emotion Recognition from Speech, developed for the CodeAlpha Machine Learning Internship program[cite: 1, 2]. The primary objective of this system is to accurately classify human emotions from raw audio speech files using digital signal processing and deep learning architectures[cite: 2]. 

The model processes human voice recordings and categorizes them into one of eight distinct emotional states: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised.

## Dataset Specification
This project utilizes the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)[cite: 2]. 
* **Source:** Kaggle / Zenodo (RAVDESS Emotional Speech Audio)
* **Format:** 16-bit, 48kHz .wav files.
* **Scope:** 24 professional actors (12 female, 12 male) vocalizing two lexically-matched statements in a neutral North American accent.
* **Emotions Captured:** 8 discrete emotions.
* **Data Split:** The system programmatically splits the real data into Training Data (80%) for model optimization and Test Data (20%) for unbiased evaluation on unseen voice samples.

## System Architecture and Methodology
The pipeline consists of three core phases: Audio Preprocessing, Feature Extraction, and Sequence Modeling.

### 1. Feature Extraction (Digital Signal Processing)
Raw audio waveforms contain extreme dimensional complexity. To make the data interpretable for the neural network, this system leverages the `librosa` library to extract Mel-Frequency Cepstral Coefficients (MFCCs) [cite: 2]. 
* **MFCCs:** The system extracts 40 sequential MFCCs from each audio file. These coefficients mathematically represent the short-term power spectrum of the sound, capturing critical vocal characteristics such as pitch, timbre, and vocal tract resonance while discarding background noise.

### 2. Deep Learning Model (LSTM)
Because speech is time-series sequence data, standard feed-forward networks are insufficient. This project implements a Long Short-Term Memory (LSTM) network built with PyTorch[cite: 2]. 
* **Architecture:** 
  - Input Layer: 40 dimensions (matching the MFCC features).
  - Hidden Layers: 2 stacked LSTM layers with 128 hidden units each, featuring a 20% Dropout rate to prevent overfitting.
  - Fully Connected Layer: A linear layer mapping the final hidden state to the 8 emotion classes.
* **Loss & Optimization:** Categorical Cross-Entropy Loss combined with the Adam optimizer (learning rate = 0.001).

## Repository Structure
Ensure your local environment matches the following directory tree before execution:

```text
CodeAlpha_SpeechEmotionRecognition/
│
├── proper_speech_emotion.py       # Main training and evaluation script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
└── ravdess_data/                  # Unzipped RAVDESS dataset
    ├── Actor_01/                  # Contains .wav files for Actor 1
    ├── Actor_02/
    ├── ...
    └── Actor_24/
