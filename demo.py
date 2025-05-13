import os
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Class definitions
CLASS_MAPPING = {0: 'five', 1: 'four', 2: 'one', 3: 'three', 4: 'two'}

# CNN model for keyword recognition
class KeywordCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(KeywordCNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 3 * 25, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# Function to extract MFCC from audio file
def extract_mfcc(path, max_len=100):
    try:
        # Load audio using soundfile
        signal, sr = sf.read(path)
        
        # Convert to mono if stereo
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)
        
        # Resample to 16000 Hz if necessary
        if sr != 16000:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        
        # Adjust MFCC length
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
            
        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return np.zeros((13, max_len), dtype=np.float32)

# Function to classify audio file
def classify_audio(model, audio_path):
    # Extract features
    mfcc = extract_mfcc(audio_path)
    
    # Visualize MFCC features
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC for {os.path.basename(audio_path)}')
    plt.tight_layout()
    plt.savefig(f"mfcc_{os.path.basename(audio_path).split('.')[0]}.png")
    plt.close()
    
    # Convert to tensor and add necessary dimensions
    x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)
    
    # Switch model to evaluation mode
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        
    # Get results
    predicted_word = CLASS_MAPPING[predicted_idx]
    confidence = probabilities[0][predicted_idx].item()
    
    print(f"\nResult for {os.path.basename(audio_path)}:")
    print(f"Recognized word: {predicted_word}")
    print(f"Confidence: {confidence:.4f}")
    
    # Output probabilities for all classes
    print("\nProbabilities for all classes:")
    for idx, cls_name in CLASS_MAPPING.items():
        prob = probabilities[0][idx].item()
        print(f"{cls_name}: {prob:.4f}")
    
    return predicted_word, confidence

def main():
    print("===== Keyword Recognition System Demo =====")
    
    # Model selection - default to best epoch
    model_path = 'keyword_spotter_model.pt.epoch9'
    if not os.path.exists(model_path):
        model_path = 'keyword_spotter_model.pt'  # Fallback option
    
    print(f"Using model: {model_path}")
    print(f"Supported keywords: {', '.join(CLASS_MAPPING.values())}")
    
    # Load model
    try:
        model = KeywordCNN(num_classes=len(CLASS_MAPPING))
        model.load_state_dict(torch.load(model_path))
        print("Model successfully loaded!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Interactive mode
    while True:
        print("\n" + "="*50)
        audio_path = input("Enter path to audio file for recognition (or 'q' to exit): ")
        
        if audio_path.lower() == 'q':
            break
        
        if not os.path.exists(audio_path):
            print(f"File {audio_path} not found. Please check the path.")
            continue
            
        try:
            classify_audio(model, audio_path)
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()