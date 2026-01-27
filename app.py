
# from flask import Flask, render_template, request, jsonify
# import cv2
# import numpy as np
# import base64
# from tensorflow import keras
# import mediapipe as mp
# from pathlib import Path

# # ------------------ Flask setup ------------------
# app = Flask(__name__)

# # ------------------ MediaPipe Hand Landmarker setup ------------------
# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# # Path to your MediaPipe Hand Landmarker task model
# HAND_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# # Create options for the hand landmarker
# hand_options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH.resolve())),
#     running_mode=VisionRunningMode.IMAGE,
#     num_hands=1
# )

# # Create hand landmarker instance
# landmarker = HandLandmarker.create_from_options(hand_options)

# # ------------------ Load pre-trained Keras model ------------------
# # Replace with the path to your new .h5 model
# MODEL_PATH = Path(__file__).parent / "asl_mediapipe_mlp_model.h5"
# model = keras.models.load_model(str(MODEL_PATH.resolve()))

# # Update LABELS to match your model classes
# LABELS = [
#     'A','B','C','D','E','F','G','H','I','J','K','L','M',
#     'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Space','Delete'
# ]

# # ------------------ Helper function to extract landmarks ------------------
# def extract_landmarks(frame):
#     """
#     Convert OpenCV frame to MediaPipe Image and extract normalized landmarks.
#     Returns a (1, 63) numpy array or None if no hand detected.
#     """
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

#     # Detect hands
#     result = landmarker.detect(mp_image)

#     if result.hand_landmarks:
#         landmark_list = []
#         # In the new Tasks API, result.hand_landmarks is a list of HandLandmark objects
#         # Each HandLandmark has a property 'landmarks' which is a list of 21 points
#         for hand in result.hand_landmarks:
#             # hand.landmarks is a list of 21 normalized landmarks
#             for lm in hand.landmarks:
#                 landmark_list.extend([lm.x, lm.y, lm.z])
#         return np.array(landmark_list).reshape(1, -1)

#     return None

# # ------------------ Flask Routes ------------------
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.json["image"]
#         header, encoded = data.split(",", 1)
#         data_bytes = base64.b64decode(encoded)
#         nparr = np.frombuffer(data_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # ------------------- EXTRACT LANDMARKS -------------------
#         landmarks = extract_landmarks(frame)

#         # ---------- DEBUGGING ----------
#         if landmarks is None:
#             print("No hands detected!")   # Check if hand is visible
#         else:
#             print("Landmarks shape:", landmarks.shape)  # Should be (1,63) for 1 hand
#         # --------------------------------

#         if landmarks is not None:
#             preds = model.predict(landmarks, verbose=0)
#             idx = int(np.argmax(preds))
#             confidence = float(np.max(preds))
#             return jsonify({
#                 "prediction": LABELS[idx],
#                 "confidence": round(confidence, 2)
#             })
#         else:
#             return jsonify({"prediction": None, "confidence": 0})

#     except Exception as e:
#         return jsonify({"prediction": None, "confidence": 0, "error": str(e)})

# # ------------------ Run App ------------------
# if __name__ == "__main__":
#     app.run(debug=True)
# ____________________________________________________________________

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow import keras
import mediapipe as mp
from pathlib import Path

# ------------------ Flask setup ------------------
app = Flask(__name__)

# ------------------ MediaPipe Hand Landmarker setup ------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to your MediaPipe Hand Landmarker task model
# Ensure 'hand_landmarker.task' is in the same folder
HAND_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Create options for the hand landmarker
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH.resolve())),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Create hand landmarker instance
landmarker = HandLandmarker.create_from_options(hand_options)

# ------------------ Load pre-trained Keras model ------------------
# Ensure your .h5 file is in the same folder
MODEL_PATH = Path(__file__).parent / "asl_mediapipe_mlp_model.h5"
model = keras.models.load_model(str(MODEL_PATH.resolve()))

# Labels matching your model
LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Space','Delete'
]

# ------------------ Helper function to extract landmarks ------------------
def extract_landmarks(frame):
    """
    Convert OpenCV frame to MediaPipe Image and extract normalized landmarks.
    Returns a (1, 63) numpy array or None if no hand detected.
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect hands
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        landmark_list = []
        
        # --- THE FIX IS HERE ---
        # result.hand_landmarks is a list of lists: [[landmark_1, landmark_2, ...]]
        # We grab the first hand (index 0) directly.
        first_hand_landmarks = result.hand_landmarks[0] 
        
        # Iterate through the 21 landmarks of the first hand
        for lm in first_hand_landmarks:
            landmark_list.extend([lm.x, lm.y, lm.z])
            
        return np.array(landmark_list).reshape(1, -1)

    return None

# ------------------ Flask Routes ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("image")
        if not data:
             return jsonify({"prediction": None, "confidence": 0})

        # Decode Base64 image
        header, encoded = data.split(",", 1)
        data_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(data_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Extract Landmarks
        landmarks = extract_landmarks(frame)

        # 2. Predict if landmarks found
        if landmarks is not None:
            preds = model.predict(landmarks, verbose=0)
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            
            # Optional: Threshold to avoid noise
            if confidence > 0.5:
                return jsonify({
                    "prediction": LABELS[idx],
                    "confidence": round(confidence, 2)
                })
            else:
                return jsonify({"prediction": None, "confidence": round(confidence, 2)})
        else:
            return jsonify({"prediction": None, "confidence": 0})

    except Exception as e:
        print(f"Server Error: {e}") # Print error to terminal for debugging
        return jsonify({"prediction": None, "confidence": 0, "error": str(e)})

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(debug=True)