from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception as e:
    print(f"⚠️ Mediapipe not available: {e}")
    mp = None
    HAS_MEDIAPIPE = False

try:
    import tensorflow as tf
    HAS_TF = True
except Exception as e:
    print(f"⚠️ TensorFlow not available: {e}")
    tf = None
    HAS_TF = False

import time
import os

app = Flask(__name__)

# === Load Model ===
MODEL_PATH = os.path.join("models", "gesture_static_model.h5")
print("🧠 Loading model...")
model = None
if HAS_TF and os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    if not HAS_TF:
        print("⚠️ TensorFlow not available — server will run without model predictions.")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH} — server will run without model predictions.")

# === Load Labels ===
try:
    label_classes = np.load("models/gesture_label_classes.npy", allow_pickle=True)
except Exception as e:
    print("⚠️ Could not load label classes:", e)
    label_classes = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G'])  # fallback
label_classes = label_classes.tolist()

# === Mediapipe Setup (optional) ===
if HAS_MEDIAPIPE:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    # === Initialize camera ===
    try:
        camera = cv2.VideoCapture(0)
    except Exception as e:
        print(f"⚠️ Could not open camera: {e}")
        camera = None
else:
    mp_hands = None
    mp_drawing = None
    hands = None
    camera = None

# --- Frame Generator ---
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        label = "Detecting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)

                prediction = model.predict(landmarks, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                label = f"{label_classes[predicted_class]} ({confidence*100:.1f}%)"

        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
@app.route('/index.html')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index: {e}")
        return str(e), 404

@app.route('/sign-detection.html')
def sign_detection():
    try:
        return render_template('sign-detection.html')
    except Exception as e:
        print(f"Error rendering sign detection: {e}")
        return str(e), 404

@app.route('/asl-alphabet.html')
def asl_alphabet():
    try:
        return render_template('asl-alphabet.html')
    except Exception as e:
        print(f"Error rendering ASL alphabet: {e}")
        return str(e), 404

@app.route('/video-call.html')
def video_call():
    try:
        return render_template('video-call.html')
    except Exception as e:
        print(f"Error rendering video call: {e}")
        return str(e), 404

@app.route('/video_feed')
def video_feed():
    if not HAS_MEDIAPIPE:
        return "Server-side MediaPipe not available. Use the browser-based detector (sign-detection.html).", 501
    if camera is None:
        return "Camera not available on server.", 503
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if data is None or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks in request'}), 400

        landmarks = np.array(data['landmarks'], dtype=np.float32).reshape(1, -1)
        prediction = model.predict(landmarks, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return jsonify({
            'gesture': label_classes[predicted_class],
            'confidence': confidence
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'labels_count': len(label_classes) if label_classes else 0
    })

if __name__ == '__main__':
    print("🚀 Running Flask server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
