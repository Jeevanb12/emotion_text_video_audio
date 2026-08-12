# Multimodal Emotion Detector (Text, Audio, Video)

A Flask web application for detecting emotions across text, audio, and video inputs using deep learning models (DistilBERT, Keras/TensorFlow CNNs, librosa, and OpenCV).

## 🚀 Features

- **Text Emotion Analysis**: Fine-tuned DistilBERT transformer for text emotion classification.
- **Audio Emotion Analysis**: Audio feature extraction (MFCCs) using Librosa and classification via Keras `.h5` model.
- **Video Emotion Analysis**: Facial emotion detection using OpenCV frame extraction and deep learning vision models.
- **Web Interface**: Interactive Flask application with template support.

## 🛠️ Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Jeevanb12/emotion-text-video-audio.git
   cd emotion_text_video_audio
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

## 📦 Large Model Files Note

Large model weights (e.g., `trnsformer.zip`, transformer checkpoints `>100MB`) are excluded via `.gitignore` to adhere to GitHub file size limits. Ensure you download or extract the required transformer model weights into `trnsformer/trnsformer/` prior to running text emotion detection.
