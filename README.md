# CNN-based Keyword Recognition System

This project is a neural network-based system for keyword recognition using Convolutional Neural Networks (CNN) and MFCC features. The system is trained to recognize five keywords: 'one', 'two', 'three', 'four', 'five'.

## Model Architecture

The system uses the following architecture:
- MFCC feature extraction from audio signals (13 coefficients)
- Convolutional neural network with two convolutional layers
- Fully connected layers for classification
## Training Results

The model was trained for 10 epochs using the Adam optimizer. The best results were achieved at epoch 9:

- **Validation accuracy**: 91.8%
- **Average F1-score**: 0.86

Performance metrics by class:

| Class  | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| five   | 0.81      | 0.86   | 0.83     |
| four   | 0.93      | 0.79   | 0.85     |
| one    | 0.82      | 0.99   | 0.90     |
| three  | 0.94      | 0.75   | 0.83     |
| two    | 0.83      | 0.90   | 0.86     |

## Demo Script

A demo script `demo.py` has been created to easily test the recognition on any audio files.
