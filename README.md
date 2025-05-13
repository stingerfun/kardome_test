CNN-based Keyword Recognition System
This project is a neural network-based system for keyword recognition using Convolutional Neural Networks (CNN) and MFCC features. The system is trained to recognize five keywords: 'one', 'two', 'three', 'four', 'five'.
Model Architecture
The system uses the following architecture:

MFCC feature extraction from audio signals (13 coefficients)
Convolutional neural network with two convolutional layers
Fully connected layers for classification

KeywordCNN(
  (conv): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0)
    (3): Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0)
  )
  (fc): Sequential(
    (0): Flatten()
    (1): Linear(in_features=2400, out_features=128)
    (2): ReLU()
    (3): Linear(in_features=128, out_features=5)
  )
)
Training Results
The model was trained for 10 epochs using the Adam optimizer. The best results were achieved at epoch 9:

Validation accuracy: 91.8%
Average F1-score: 0.86

Performance metrics by class:
ClassPrecisionRecallF1-Scorefive0.810.860.83four0.930.790.85one0.820.990.90three0.940.750.83two0.830.900.86
