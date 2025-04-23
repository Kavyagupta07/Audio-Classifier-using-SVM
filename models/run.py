# predict_audio.py
import os
import librosa
import numpy as np
from joblib import load
from pydub import AudioSegment

# Configuration
MODEL_PATH = "models/audio_classifier.joblib"
SCALER_PATH = "models/scaler.joblib"
CLASSES = ["speech", "music", "noise"]  # Must match training classes

# Load model and scaler
try:
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def convert_to_wav_if_needed(file_path):
    """Convert MP3 to WAV if needed (librosa works better with WAV)"""
    if file_path.lower().endswith('.mp3'):
        try:
            audio = AudioSegment.from_mp3(file_path)
            wav_path = file_path.replace('.mp3', '_converted.wav')
            audio.export(wav_path, format="wav")
            print(f"Converted MP3 to WAV: {wav_path}")
            return wav_path
        except Exception as e:
            print(f"MP3 conversion failed: {e}")
    return file_path

def extract_features(file_path):
    """Extract features matching the training setup"""
    try:
        file_path = convert_to_wav_if_needed(file_path)
        X, sample_rate = librosa.load(file_path, sr=None, duration=3)
        result = []
        
        # MFCC (40 coefficients)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result.extend(mfccs)
        
        # Chroma features
        stft = np.abs(librosa.stft(X))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result.extend(chroma)
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result.extend(mel)
        
        return np.array(result)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_audio(file_path):
    """Make prediction on a single audio file"""
    features = extract_features(file_path)
    if features is None:
        return "Error: Could not extract features"
    
    # Reshape and scale features
    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)
    
    # Predict
    class_idx = model.predict(scaled_features)[0]
    confidence = np.max(model.predict_proba(scaled_features))
    
    return CLASSES[class_idx], confidence

def predict_folder(folder_path):
    """Predict all audio files in a folder"""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    print(f"\nPredicting files in: {folder_path}")
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.wav', '.mp3')):
            continue
            
        file_path = os.path.join(folder_path, file_name)
        prediction, confidence = predict_audio(file_path)
        print(f"{file_name}: {prediction} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    print("Audio Classifier Prediction")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Predict single audio file")
        print("2. Predict all files in a folder")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            file_path = input("Enter audio file path: ").strip('"')
            if os.path.exists(file_path):
                prediction, confidence = predict_audio(file_path)
                print(f"\nPrediction: {prediction} (confidence: {confidence:.2f})")
            else:
                print("File not found")
                
        elif choice == "2":
            folder_path = input("Enter folder path: ").strip('"')
            predict_folder(folder_path)
            
        elif choice == "3":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice")