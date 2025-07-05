# Main FastAPI application file
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import json
import numpy as np
import cv2
import mediapipe as mp
import shutil
import os
from typing import List

# Import necessary components from the codebase (adaptations might be needed)
# from data.ds_2 import Preprocessing as LandmarkPreprocessor # If directly usable

app = FastAPI()

# --- Configuration & Model Loading ---
MODEL_PATH = "datamount/weights/cfg_2/fold-1/model.tflite"
INFERENCE_ARGS_PATH = "datamount/weights/cfg_2/fold-1/inference_args.json"
CHARACTER_MAP_PATH = "datamount/character_to_prediction_index.json"

# Templates and static files
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static") # If you have static files

# Load character map and create reverse map
try:
    with open(CHARACTER_MAP_PATH, 'r') as f:
        character_to_num = json.load(f)

    pad_token = 'P'
    start_token = 'S'
    end_token = 'E'

    n_chars = len(character_to_num)
    character_to_num[pad_token] = n_chars
    character_to_num[start_token] = n_chars + 1
    character_to_num[end_token] = n_chars + 2

    num_to_character = {v: k for k, v in character_to_num.items()}

    PAD_TOKEN_ID = character_to_num[pad_token]
    START_TOKEN_ID = character_to_num[start_token]
    END_TOKEN_ID = character_to_num[end_token]

    print("Character map loaded successfully.")
except Exception as e:
    print(f"Error loading character map: {e}")
    character_to_num = {}
    num_to_character = {}
    PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID = -1, -1, -1

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TensorFlow Lite model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None

# Load inference arguments (for selected_columns)
try:
    with open(INFERENCE_ARGS_PATH, 'r') as f:
        inference_args = json.load(f)
    SELECTED_COLUMNS = inference_args.get('selected_columns', [])
    if not SELECTED_COLUMNS:
        print("Warning: 'selected_columns' not found. Landmark extraction might be incorrect.")
    print("Inference arguments loaded successfully.")
except Exception as e:
    print(f"Error loading inference arguments: {e}")
    inference_args = {}
    SELECTED_COLUMNS = []

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Preprocessing, Postprocessing (Adapted from previous Flask app.py) ---
MAX_LEN_FRAMES = 384
N_LANDMARKS_EXPECTED = 130

def extract_landmarks_from_frame(image_rgb: np.ndarray) -> List[float]:
    results = holistic_model.process(image_rgb)
    frame_landmarks = {}
    def get_landmarks(holistic_landmarks, landmark_type_name):
        lm_dict = {}
        if holistic_landmarks:
            for i, lm in enumerate(holistic_landmarks.landmark):
                lm_dict[f'{landmark_type_name}_{i}_x'] = lm.x
                lm_dict[f'{landmark_type_name}_{i}_y'] = lm.y
                lm_dict[f'{landmark_type_name}_{i}_z'] = lm.z
        return lm_dict

    frame_landmarks.update(get_landmarks(results.face_landmarks, 'face'))
    frame_landmarks.update(get_landmarks(results.pose_landmarks, 'pose'))
    frame_landmarks.update(get_landmarks(results.left_hand_landmarks, 'left_hand'))
    frame_landmarks.update(get_landmarks(results.right_hand_landmarks, 'right_hand'))

    ordered_landmarks = [frame_landmarks.get(col, np.nan) for col in SELECTED_COLUMNS]
    return ordered_landmarks

def preprocess_video_frames(video_file_path: str):
    cap = cv2.VideoCapture(video_file_path)
    frames_landmarks_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_for_frame = extract_landmarks_from_frame(image_rgb)
        frames_landmarks_list.append(landmarks_for_frame)
    cap.release()

    if not frames_landmarks_list: return None, None

    landmarks_np = np.array(frames_landmarks_list, dtype=np.float32)

    expected_features = N_LANDMARKS_EXPECTED * 3
    if landmarks_np.shape[1] != expected_features:
        if landmarks_np.shape[1] < expected_features:
            padding = np.full((landmarks_np.shape[0], expected_features - landmarks_np.shape[1]), np.nan)
            landmarks_np = np.concatenate([landmarks_np, padding], axis=1)
        else:
            landmarks_np = landmarks_np[:, :expected_features]

    landmarks_np = landmarks_np.reshape(landmarks_np.shape[0], N_LANDMARKS_EXPECTED, 3)

    mean = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)
    for i in range(3):
        channel_data = landmarks_np[:, :, i]
        nonan_channel_data = channel_data[~np.isnan(channel_data)]
        if nonan_channel_data.size > 0:
            mean[i] = np.mean(nonan_channel_data)
            std[i] = np.std(nonan_channel_data)
            if std[i] == 0: std[i] = 1.0
        else: mean[i] = 0.0; std[i] = 1.0
    landmarks_np = (landmarks_np - mean[None, None, :]) / std[None, None, :]
    landmarks_np[np.isnan(landmarks_np)] = 0.0

    current_len = landmarks_np.shape[0]
    processed_frames = np.zeros((MAX_LEN_FRAMES, N_LANDMARKS_EXPECTED, 3), dtype=np.float32)
    input_mask = np.zeros(MAX_LEN_FRAMES, dtype=np.float32)

    if current_len > MAX_LEN_FRAMES:
        temp_reshaped = landmarks_np.reshape(current_len, -1)
        resized_data = cv2.resize(temp_reshaped, (N_LANDMARKS_EXPECTED * 3, MAX_LEN_FRAMES), interpolation=cv2.INTER_LINEAR)
        processed_frames = resized_data.reshape(MAX_LEN_FRAMES, N_LANDMARKS_EXPECTED, 3)
        input_mask[:] = 1.0
    else:
        processed_frames[:current_len, :, :] = landmarks_np
        input_mask[:current_len] = 1.0

    # Ensure correct dtype for TFLite model
    processed_frames_dtype = processed_frames.astype(input_details[0]['dtype'])
    input_mask_dtype = input_mask.astype(input_details[1]['dtype'] if len(input_details) > 1 else np.float32)

    return processed_frames_dtype, input_mask_dtype

def postprocess_output(model_output_sequence: np.ndarray) -> str:
    char_list = []
    for token_id in model_output_sequence:
        if token_id == END_TOKEN_ID: break
        if token_id == START_TOKEN_ID or token_id == PAD_TOKEN_ID: continue
        char_list.append(num_to_character.get(int(token_id), '?')) # Ensure token_id is int for dict lookup
    return "".join(char_list)

# --- FastAPI Endpoints ---
@app.get("/")
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_video(request: Request, file: UploadFile = File(...)):
    if interpreter is None:
        return templates.TemplateResponse("result.html", {"request": request, "prediction": "Error: Model not loaded."})

    temp_video_path = f"temp_{file.filename}"
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processed_input_data, input_mask_data = preprocess_video_frames(temp_video_path)

        if processed_input_data is None:
            return templates.TemplateResponse("result.html", {"request": request, "prediction": "Error: Could not process video."})

        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(processed_input_data, axis=0))
        if len(input_details) > 1:
            if input_details[1]['shape'].tolist() == [1, MAX_LEN_FRAMES]: # Check mask shape compatibility
                 interpreter.set_tensor(input_details[1]['index'], np.expand_dims(input_mask_data, axis=0))
            else:
                print(f"Warning: TFLite model's second input shape {input_details[1]['shape']} doesn't match expected mask shape {[1, MAX_LEN_FRAMES]}.")


        interpreter.invoke()
        raw_output_sequence = interpreter.get_tensor(output_details[0]['index'])

        model_output_sequence = raw_output_sequence[0] if raw_output_sequence.ndim > 1 and raw_output_sequence.shape[0] == 1 else raw_output_sequence

        prediction_text = postprocess_output(model_output_sequence)

    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}\n{traceback.format_exc()}")
        prediction_text = f"Error: {e}"
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        await file.close() # Ensure the UploadFile resource is closed

    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction_text})

if __name__ == "__main__":
    import uvicorn
    # This part is for local execution.
    # For deployment, you'd typically run: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
