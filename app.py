# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import numpy as np
from joblib import load
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Load trained model and scaler
# model = load('dataset\models\audio_classifier.joblib')
model = load('models/audio_classifier.joblib')
# scaler = load('dataset\models\scaler.joblib')
scaler = load('models/scaler.joblib')
CLASSES = ["speech", "music", "noise"]

# Feature extraction (same as training)
def extract_features(file_path):
    try:
        X, sample_rate = librosa.load(file_path, sr=None, duration=3)
        result = []
        
        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result.extend(mfccs)
        
        # Chroma
        stft = np.abs(librosa.stft(X))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result.extend(chroma)
        
        # Mel
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result.extend(mel)
        
        return np.array(result)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
            
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
            
#         if file:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
            
#             # Extract features and predict
#             features = extract_features(filepath)
#             if features is not None:
#                 features = features.reshape(1, -1)
#                 scaled_features = scaler.transform(features)
#                 prediction = model.predict(scaled_features)[0]
#                 result = CLASSES[prediction]
#                 confidence = np.max(model.predict_proba(scaled_features))
                
#                 # Clean up uploaded file
#                 os.remove(filepath)
                
#                 return render_template('result.html', 
#                                       result=result.upper(),
#                                       confidence=f"{confidence*100:.1f}%",
#                                       filename=filename)
            
#             os.remove(filepath)
#             return render_template('error.html', message="Error processing audio file")
    
#     return render_template('index.html')






@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract features and predict
            features = extract_features(filepath)
            if features is not None:
                features = features.reshape(1, -1)
                scaled_features = scaler.transform(features)
                prediction = model.predict(scaled_features)[0]
                result = CLASSES[prediction]
                confidence = "N/A"  # You can display "N/A" or a default value here since we're not using predict_proba
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return render_template('result.html', 
                                      result=result.upper(),
                                      confidence=confidence,
                                      filename=filename)
            
            os.remove(filepath)
            return render_template('error.html', message="Error processing audio file")
    
    return render_template('index.html')

    # return render_template('result.html', 
                       # result=result.upper(),
                       # confidence=f"{confidence*100:.1f}%",
                       # filename=filename)
    # return render_template('error.html', message="Error processing audio file")

# HTML Templates with Dark Theme
app.jinja_env.globals.update(
    index_html=lambda: '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Classifier</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #2a2a2a; color: #e0e0e0; }
            .container { max-width: 600px; margin-top: 50px; }
            .card { background-color: #3a3a3a; border: 1px solid #4a4a4a; }
            .btn-primary { background-color: #0069d9; border-color: #0062cc; }
            .result-card { background-color: #4a4a4a; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card p-4">
                <h1 class="text-center mb-4">Audio Classification</h1>
                <form method="post" enctype="multipart/form-data">
                    <div class="mb-3">
                        <input type="file" class="form-control" name="file" accept=".wav,.mp3" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Analyze Audio</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    ''',
    result_html=lambda result, confidence, filename: f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Result</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #2a2a2a; color: #e0e0e0; }}
            .container {{ max-width: 600px; margin-top: 50px; }}
            .card {{ background-color: #3a3a3a; border: 1px solid #4a4a4a; }}
            .result-card {{ background-color: {'#28a74530' if result == 'SPEECH' else '#007bff30' if result == 'MUSIC' else '#dc354530'}; }}
            .back-btn {{ background-color: #0069d9; border-color: #0062cc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card p-4 result-card">
                <h2 class="text-center mb-4">Analysis Result</h2>
                <div class="text-center mb-3">
                    <h4>File: {filename}</h4>
                    <div class="display-4 mb-3">{result}</div>
                    <div class="h5">Confidence: {confidence}</div>
                </div>
                <a href="/" class="btn back-btn w-100">Analyze Another File</a>
            </div>
        </div>
    </body>
    </html>
    ''',
    error_html=lambda message: f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #2a2a2a; color: #e0e0e0; }}
            .container {{ max-width: 600px; margin-top: 50px; }}
            .card {{ background-color: #3a3a3a; border: 1px solid #4a4a4a; }}
            .error-card {{ background-color: #dc354530; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card p-4 error-card">
                <h2 class="text-center text-danger mb-4">Error</h2>
                <p class="text-center">{message}</p>
                <a href="/" class="btn btn-danger w-100">Try Again</a>
            </div>
        </div>
    </body>
    </html>
    '''
)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
