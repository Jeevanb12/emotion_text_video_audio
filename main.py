from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
import os
import tempfile
import pickle
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
import speech_recognition as sr
import torch
from collections import Counter
import transformers, sys
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
import sounddevice as sd
import wave
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
audio_model = None
audio_label_encoder = None
text_model = None
text_tokenizer = None
text_device = None
text_label_map = None
video_model = None
video_reverse_map = None

def load_audio_model():
    global audio_model, audio_label_encoder
    audio_model = load_model("audio_emotion.h5")
    with open("label_encoder.pkl", "rb") as f:
        audio_label_encoder = pickle.load(f)

def load_text_model():
    global text_model, text_tokenizer, text_device, text_label_map
    # Absolute path to your model folder
    model_dir = r"C:\Users\jeeva\Downloads\emotion_text_video_audio\trnsformer\trnsformer"

    # Load tokenizer + model
    text_tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    text_model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    # Move to device
    text_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model.to(text_device)

    # Your custom label mapping
    text_label_map = {
        0: "sad",
        1: "happy",
        2: "love",
        3: "angry",
        4: "fear",
        5: "surprise"
    }

def load_video_model():
    global video_model, video_reverse_map
    video_model = load_model("video_emotion.h5")
    video_reverse_map = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }

# Your exact original functions
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.85):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data, sample_rate):
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    return mfcc

def transform_audio(data, fns, sample_rate):
    fn = random.choice(fns)
    if fn == pitch:
        return fn(data, sample_rate)
    elif fn == "None":
        return data
    elif fn in [noise, stretch]:
        return fn(data)
    else:
        return data

def get_features(audio_path):
    data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
    fns = [noise, pitch, "None"]
    features_list = []
    for _ in range(3):
        data_aug = transform_audio(data, fns, sample_rate)
        mfcc = extract_features(data_aug, sample_rate)
        mfcc = mfcc[:, :108]
        features_list.append(mfcc)
    return features_list

def predict_audio_emotion(audio_path, audio_model, audio_label_encoder):
    features = get_features(audio_path)
    predictions = []
    for feat in features:
        feat = np.expand_dims(feat, axis=0)
        feat = np.expand_dims(feat, axis=3)
        feat = np.swapaxes(feat, 1, 2)
        pred = audio_model.predict(feat)
        predictions.append(pred)
    avg_pred = np.mean(predictions, axis=0)
    emotion = audio_label_encoder.inverse_transform(avg_pred)[0]
    return emotion[0]

def predict_text_emotion(text, text_model, text_tokenizer, text_device, text_label_map):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(text_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
        logits = outputs.logits
    pred = logits.argmax(dim=-1).item()
    return text_label_map.get(pred, "Unknown")

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio_data)
    except Exception as e:
        transcription = f"Transcription failed: {e}"
    return transcription

def extract_audio(video_path, output_audio_path="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, logger=None)
    return output_audio_path

def extract_video_sequence(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    if total_frames < num_frames:
        indices = range(total_frames - 1)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames_dict = {}
    frame_idx = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            frames_dict[frame_idx] = normalized
        frame_idx += 1
    cap.release()
    try:
        seq = [frames_dict[i] for i in indices]
    except KeyError:
        return None
    seq = np.array(seq)
    seq = np.expand_dims(seq, axis=-1)
    return seq

def predict_video_emotion(video_path, video_model, video_reverse_map):
    seq = extract_video_sequence(video_path, num_frames=100000)
    if seq is None:
        return "Unknown"

    smaller_sequences = [np.expand_dims(seq[i:i + 10], axis=0) for i in range(0, len(seq), 10)]
    predictions = []

    for smaller_seq in smaller_sequences:
        pred = video_model.predict(smaller_seq)
        predicted_class = np.argmax(pred, axis=1)[0]
        predictions.append(predicted_class)

    most_common_prediction = Counter(predictions).most_common(1)[0][0]
    return video_reverse_map.get(most_common_prediction, "Unknown")

# def weighted_voting(audio_emotion, video_emotion, text_emotion, audio_weight=2, video_weight=1, text_weight=1):
#     all_possible_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'Unknown']
#     weights = {emotion: 0 for emotion in all_possible_emotions}
#     weights[audio_emotion] += audio_weight
#     weights[video_emotion] += video_weight
#     weights[text_emotion] += text_weight
#     max_weight = max(weights.values())
#     max_emotions = [emotion for emotion, weight in weights.items() if weight == max_weight]
#     if len(max_emotions) > 1:
#         if audio_emotion in max_emotions:
#             return video_emotion
#     final_prediction = max(weights, key=weights.get)
#     return final_prediction

def weighted_voting(audio_emotion, video_emotion, text_emotion, audio_weight=2, video_weight=1, text_weight=1):
    all_possible_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'Unknown']
    weights = {emotion: 0 for emotion in all_possible_emotions}
    
    # Only add weights for non-None emotions
    if audio_emotion and audio_emotion in weights:
        weights[audio_emotion] += audio_weight
    if video_emotion and video_emotion in weights:
        weights[video_emotion] += video_weight
    if text_emotion and text_emotion in weights:
        weights[text_emotion] += text_weight
    
    max_weight = max(weights.values())
    max_emotions = [emotion for emotion, weight in weights.items() if weight == max_weight]
    
    if len(max_emotions) > 1:
        # Return the first non-None emotion from the tie
        if audio_emotion and audio_emotion in max_emotions:
            return audio_emotion
        if video_emotion and video_emotion in max_emotions:
            return video_emotion
        if text_emotion and text_emotion in max_emotions:
            return text_emotion
    
    final_prediction = max(weights, key=weights.get)
    return final_prediction
def record_audio(duration=5, sample_rate=44100):
    """Record audio from microphone"""
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        
        # Save to temporary file
        temp_path = f"temp_recording_{int(time.time())}.wav"
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        return temp_path
    except Exception as e:
        return None

# Emotion icons
EMOTION_ICONS = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'surprise': '😲',
    'disgust': '🤢',
    'neutral': '😐',
    'calm': '😌',
    'love': '😍',
    'Unknown': '❓'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'})
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract audio
        audio_temp_path = "temp_audio.wav"
        extract_audio(filepath, audio_temp_path)

        # Get transcription
        transcription = transcribe_audio(audio_temp_path)

        # Predict emotions
        audio_emotion = predict_audio_emotion(audio_temp_path, audio_model, audio_label_encoder)
        text_emotion = predict_text_emotion(transcription, text_model, text_tokenizer, text_device, text_label_map)
        video_emotion = predict_video_emotion(filepath, video_model, video_reverse_map)

        # Final prediction
        final_prediction = weighted_voting(audio_emotion, video_emotion, text_emotion)
        
        # Cleanup
        try:
            os.remove(audio_temp_path)
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'final_prediction': final_prediction,
            'emotion_icon': EMOTION_ICONS.get(final_prediction, '❓'),
            'transcription': transcription if not transcription.startswith("Transcription failed") else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# @app.route('/analyze_audio', methods=['POST'])
# def analyze_audio():
#     try:
#         if 'audio' not in request.files:
#             return jsonify({'error': 'No audio file uploaded'})
        
#         file = request.files['audio']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'})
        
#         # Save uploaded file
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Get transcription and emotion
#         transcription = transcribe_audio(filepath)
#         audio_emotion = predict_audio_emotion(filepath, audio_model, audio_label_encoder)
#         text_emotion = predict_text_emotion(transcription, text_model, text_tokenizer, text_device, text_label_map)
        
#         # Final prediction (audio + text)
#         final_prediction = weighted_voting(audio_emotion, None, text_emotion, audio_weight=3, text_weight=2)
        
#         # Cleanup
#         try:
#             os.remove(filepath)
#         except:
#             pass
        
#         return jsonify({
#             'final_prediction': final_prediction,
#             'emotion_icon': EMOTION_ICONS.get(final_prediction, '❓'),
#             'transcription': transcription if not transcription.startswith("Transcription failed") else None
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)})
@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'})
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Debug: Saved audio file to {filepath}")
        
        # Get transcription and emotion
        transcription = transcribe_audio(filepath)
        print(f"Debug: Transcription: {transcription}")
        
        audio_emotion = predict_audio_emotion(filepath, audio_model, audio_label_encoder)
        print(f"Debug: Audio emotion: {audio_emotion}")
        
        text_emotion = predict_text_emotion(transcription, text_model, text_tokenizer, text_device, text_label_map)
        print(f"Debug: Text emotion: {text_emotion}")
        
        # Final prediction (audio + text)
        final_prediction = weighted_voting(audio_emotion, None, text_emotion, audio_weight=3, text_weight=2)
        print(f"Debug: Final prediction: {final_prediction}")
        
        # Cleanup
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'final_prediction': final_prediction,
            'emotion_icon': EMOTION_ICONS.get(final_prediction, '❓'),
            'transcription': transcription if not transcription.startswith("Transcription failed") else None
        })
        
    except Exception as e:
        print(f"Debug: Error in analyze_audio route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'})
        
        # Predict emotion
        text_emotion = predict_text_emotion(text, text_model, text_tokenizer, text_device, text_label_map)
        
        return jsonify({
            'final_prediction': text_emotion,
            'emotion_icon': EMOTION_ICONS.get(text_emotion, '❓'),
            'word_count': len(text.split()),
            'char_count': len(text)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/record_audio')
def record_audio_route():
    try:
        duration = int(request.args.get('duration', 5))
        audio_path = record_audio(duration)
        
        if not audio_path:
            return jsonify({'error': 'Failed to record audio'})
        
        # Get transcription and emotion
        transcription = transcribe_audio(audio_path)
        audio_emotion = predict_audio_emotion(audio_path, audio_model, audio_label_encoder)
        text_emotion = predict_text_emotion(transcription, text_model, text_tokenizer, text_device, text_label_map)
        
        # Final prediction (audio + text)
        final_prediction = weighted_voting(audio_emotion, None, text_emotion, audio_weight=3, text_weight=2)
        
        # Cleanup
        try:
            os.remove(audio_path)
        except:
            pass
        
        return jsonify({
            'final_prediction': final_prediction,
            'emotion_icon': EMOTION_ICONS.get(final_prediction, '❓'),
            'transcription': transcription if not transcription.startswith("Transcription failed") else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Loading models...")
    load_audio_model()
    load_text_model()
    load_video_model()
    print("Models loaded successfully!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)