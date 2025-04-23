# train_audio_classifier.py
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Configuration - UPDATED PATH HANDLING
current_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(current_dir)  # Points to current directory (your dataset folder)
CLASSES = ["speech", "music", "noise"]  # Must match your folder names exactly

print(f"Current directory: {current_dir}")
print(f"Looking for dataset in: {DATASET_PATH}")

# Feature extraction function
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    try:
        X, sample_rate = librosa.load(file_path, sr=None, duration=3)
        result = []
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result.extend(mfccs)
            
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result.extend(chroma)
            
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result.extend(mel)
            
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_dataset():
    features = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_PATH, class_name)
        
        if not os.path.exists(class_dir):
            print(f"\nERROR: Missing directory - {class_dir}")
            print("Please create this directory and add audio files")
            continue
            
        print(f"\nProcessing {class_name} files from: {class_dir}")
        
        file_count = 0
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if not os.path.isfile(file_path):
                continue
                
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(class_idx)
                file_count += 1
                
        print(f"Found {file_count} valid {class_name} files")
                
    return np.array(features), np.array(labels)

def train_and_save_model():
    # Verify class directories exist
    missing_dirs = [c for c in CLASSES if not os.path.exists(os.path.join(DATASET_PATH, c))]
    if missing_dirs:
        raise FileNotFoundError(
            f"Missing required directories: {missing_dirs}\n"
            f"Please create these folders in: {DATASET_PATH}"
        )
    
    # Load dataset
    print("\nLoading dataset and extracting features...")
    X, y = load_dataset()
    
    if len(X) == 0:
        raise ValueError(
            "No features extracted!\n"
            "Possible causes:\n"
            "1. No audio files in the class directories\n"
            "2. File permission issues\n"
            "3. Unsupported file formats\n"
            f"Checked directories: {[os.path.join(DATASET_PATH, c) for c in CLASSES]}"
        )
        
    # Split and scale dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    print("\nTraining SVM model...")
    model = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Save models
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "audio_classifier.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")
    dump(model, model_path)
    dump(scaler, scaler_path)
    print(f"\nModel saved to: {os.path.abspath(model_path)}")
    print(f"Scaler saved to: {os.path.abspath(scaler_path)}")

if __name__ == "__main__":
    print("Audio Classifier Training Script")
    print("=" * 50)
    train_and_save_model()