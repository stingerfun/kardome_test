import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import soundfile as sf  # Using soundfile for audio loading

# Set output paths for model, logs, and visualization
LOG_FILE = "training.log"
PLOTS_DIR = "plots"
MODEL_PATH = "keyword_spotter_model.pt"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Simple logging function that writes to both console and file
def log(message):
    """
    Log a message to both console and log file with timestamp
    """
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")
    print(message)

# Load and prepare metadata from CSV file
log("Loading metadata")
metadata = pd.read_csv("metadata.csv")
# Convert file paths to OS-specific format
metadata["filepath"] = metadata["filepath"].apply(lambda x: os.path.join("audio", x.replace('\\', os.sep)))

# Encode class labels to numerical indices
log("Encoding labels")
le = LabelEncoder()
metadata["label_idx"] = le.fit_transform(metadata["label"])

# Feature extraction function - MFCC (Mel-frequency cepstral coefficients)
def extract_mfcc(path, max_len=100):
    """
    Extract MFCC features from audio file
    
    Args:
        path (str): Path to audio file
        max_len (int): Maximum length of MFCC features (padded or truncated)
        
    Returns:
        numpy.ndarray: MFCC features with shape (13, max_len)
    """
    try:
        # Using soundfile instead of librosa.load to avoid dependencies issues
        signal, sr = sf.read(path)
        
        # Convert stereo to mono if needed
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)
        
        # Resample to 16000 Hz if needed
        if sr != 16000:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc.astype(np.float32)
    except Exception as e:
        log(f"Error processing file {path}: {e}")
        # Return zero matrix in case of an error
        return np.zeros((13, max_len), dtype=np.float32)

# Custom Dataset class for handling audio data
class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio files with MFCC features
    """
    def __init__(self, df):
        """
        Initialize dataset from dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with filepath and label_idx columns
        """
        self.df = df
        self.X = []
        self.y = df["label_idx"].values
        
        # Extract features from all files with progress tracking
        log(f"Extracting features from {len(df)} files...")
        for i, fp in enumerate(df["filepath"]):
            if i % 100 == 0:
                log(f"Processed {i}/{len(df)} files")
            self.X.append(extract_mfcc(fp))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Add channel dimension for CNN (batch_size, channels, height, width)
        return torch.tensor(self.X[idx]).unsqueeze(0), torch.tensor(self.y[idx])

# Split data into training and testing sets
train_df = metadata[metadata["type"] == "train"]
test_df = metadata[metadata["type"] == "test"]

log(f"Training set: {len(train_df)} samples")
log(f"Testing set: {len(test_df)} samples")

# Create datasets
log("Preparing datasets...")
train_dataset = AudioDataset(train_df)
test_dataset = AudioDataset(test_df)

# Determine optimal number of workers for DataLoader
# Safer approach - use 4 workers or the number of CPUs if less than 4
import multiprocessing
num_workers = min(4, multiprocessing.cpu_count())
log(f"Using {num_workers} workers for data loading")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

# CNN model architecture
class KeywordCNN(nn.Module):
    """
    Convolutional Neural Network for keyword recognition
    """
    def __init__(self, num_classes):
        """
        Initialize model
        
        Args:
            num_classes (int): Number of output classes
        """
        super(KeywordCNN, self).__init__()
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # First convolutional layer
            nn.ReLU(),                                  # Activation function
            nn.MaxPool2d(2),                            # Max pooling
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Second convolutional layer
            nn.ReLU(),                                  # Activation function
            nn.MaxPool2d(2),                            # Max pooling
        )
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),                               # Flatten for fully connected layer
            nn.Linear(32 * 3 * 25, 128),                # First fully connected layer
            nn.ReLU(),                                  # Activation function
            nn.Linear(128, num_classes),                # Output layer
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv(x)
        return self.fc(x)

# Initialize model, loss function, and optimizer
device = torch.device("cpu")
model = KeywordCNN(num_classes=len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store training metrics
train_losses = []
val_accuracies = []

# Training loop
log("Starting training")
for epoch in range(10):
    # Training phase
    model.train()
    running_loss = 0.0
    
    # Track batch progress
    batch_count = len(train_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if batch_idx % 10 == 0:
            log(f"Epoch {epoch+1}, batch {batch_idx}/{batch_count}")
            
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation phase
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, axis=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculate validation accuracy
    acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(acc)
    log(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Val Acc: {acc:.4f}")

    # Save model checkpoint after each epoch
    torch.save(model.state_dict(), f"{MODEL_PATH}.epoch{epoch+1}")
    log(f"Checkpoint saved after epoch {epoch+1}")

# Visualize training metrics
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.title("Training Loss and Validation Accuracy")
plt.savefig(os.path.join(PLOTS_DIR, "training_metrics.png"))

# Create and save confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()  # To ensure labels and title fit well
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))

# Save final model
torch.save(model.state_dict(), MODEL_PATH)
log("Training complete, model and plots saved.")

# Output class mapping information
class_mapping = {idx: label for idx, label in enumerate(le.classes_)}
log(f"Class mapping: {class_mapping}")

# Generate detailed classification report
from sklearn.metrics import classification_report
report = classification_report(all_labels, all_preds, target_names=le.classes_)
log("Classification Report:\n" + report)