from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import mediapipe as mp
import keras
app = Flask(__name__)

# 1. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 2. Load your model
# Ensure your model was trained on 63 landmarks (21 points * x,y,z)
model = keras.models.load_model('sign_language_model.h5')
LABELS = ['A', 'B', 'C', 'Space', 'Clear'] # Update to match your training classes

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    landmark_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                # Store x, y, and z coordinates
                landmark_list.extend([lm.x, lm.y, lm.z])
        return np.array(landmark_list).reshape(1, -1)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    header, encoded = data.split(",", 1)
    data_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(data_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    landmarks = extract_landmarks(frame)
    
    if landmarks is not None:
        # Prediction
        prediction = model.predict(landmarks, verbose=0)
        res_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        if confidence > 0.7: # Only return if model is sure
            return jsonify({'prediction': LABELS[res_index]})
    
    return jsonify({'prediction': None})

if __name__ == '__main__':
    app.run(debug=True)