# Main Flask application file
from flask import Flask, request, render_template
import tensorflow as tf
import json
import numpy as np
# We will need to import functions for preprocessing and postprocessing
import cv2
import mediapipe as mp

# Import necessary components from the codebase
# Assuming ds_2.py contains the relevant Preprocessing class and utility functions
# We need to adapt these for single inference rather than batch processing in training
from data.ds_2 import Preprocessing as LandmarkPreprocessor
# The original Preprocessing class in ds_2.py expects PyTorch tensors.
# We'll need to adapt it or extract its logic for NumPy arrays if using OpenCV directly.

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = "datamount/weights/cfg_2/fold-1/model.tflite"
INFERENCE_ARGS_PATH = "datamount/weights/cfg_2/fold-1/inference_args.json"
CHARACTER_MAP_PATH = "datamount/character_to_prediction_index.json"

# Load character map and create reverse map
try:
    with open(CHARACTER_MAP_PATH, 'r') as f:
        character_to_num = json.load(f)

    # Add special tokens based on cfg_2.py inspection
    # These values might differ if cfg_2.py was changed for the exported model
    # For simplicity, we'll use the values directly found in cfg_2.py for now.
    # It's safer if these are part of inference_args.json or a separate config for the app.
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


# Load the TFLite model and allocate tensors.
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TensorFlow Lite model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None

# Load inference arguments
try:
    with open(INFERENCE_ARGS_PATH, 'r') as f:
        inference_args = json.load(f)
    SELECTED_COLUMNS = inference_args.get('selected_columns', [])
    if not SELECTED_COLUMNS:
        print("Warning: 'selected_columns' not found in inference_args.json. Landmark extraction might be incorrect.")
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

# --- Preprocessing and Postprocessing ---
# Adapted from ds_2.py and train.py logic
MAX_LEN_FRAMES = 384 # From cfg_2.py (cfg.max_len)
# N_LANDMARKS should be derived from selected_columns, typically len(selected_columns) / 3
N_LANDMARKS_EXPECTED = 130 # from cfg_2.py (cfg.n_landmarks)
# The actual number of columns for x, y, z for each landmark type
# FACE: 468 landmarks * 3 = 1404 values
# POSE: 33 landmarks * 3 = 99 values
# LEFT_HAND: 21 landmarks * 3 = 63 values
# RIGHT_HAND: 21 landmarks * 3 = 63 values

# Create a mapping from landmark names to their indices in the MediaPipe output
# This is a simplified example; the actual competition `selected_columns` would be more specific.
# For now, we'll try to extract all available and then select/order them.
landmark_types = ['face', 'left_hand', 'pose', 'right_hand']
xyz_map = { 'x':0, 'y':1, 'z':2 }

# This Preprocessor is from ds_2.py, it expects PyTorch tensors.
# We will adapt its logic for numpy arrays.
# landmark_preprocessor = LandmarkPreprocessor() # Original preprocessor

def extract_landmarks_from_frame(image_rgb):
    """Extracts landmarks from a single RGB image using MediaPipe Holistic."""
    results = holistic_model.process(image_rgb)

    # Initialize a dictionary to hold all landmark data for the frame
    frame_landmarks = {}

    # Helper to extract landmarks
    def get_landmarks(holistic_landmarks, landmark_type):
        lm_dict = {}
        if holistic_landmarks:
            for i, lm in enumerate(holistic_landmarks.landmark):
                lm_dict[f'{landmark_type}_{i}_x'] = lm.x
                lm_dict[f'{landmark_type}_{i}_y'] = lm.y
                lm_dict[f'{landmark_type}_{i}_z'] = lm.z
        return lm_dict

    frame_landmarks.update(get_landmarks(results.face_landmarks, 'face'))
    frame_landmarks.update(get_landmarks(results.pose_landmarks, 'pose'))
    frame_landmarks.update(get_landmarks(results.left_hand_landmarks, 'left_hand'))
    frame_landmarks.update(get_landmarks(results.right_hand_landmarks, 'right_hand'))

    # Create a flat list of landmark values, ordered by SELECTED_COLUMNS
    # If a landmark in SELECTED_COLUMNS is not found from mediapipe, fill with NaN
    ordered_landmarks = [frame_landmarks.get(col, np.nan) for col in SELECTED_COLUMNS]

    return ordered_landmarks


def preprocess_video_frames(video_file_path):
    """
    Reads a video, extracts landmarks from each frame, normalizes, and pads/truncates.
    Returns a NumPy array suitable for the TFLite model.
    """
    cap = cv2.VideoCapture(video_file_path)
    frames_landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_for_frame = extract_landmarks_from_frame(image_rgb)
        frames_landmarks_list.append(landmarks_for_frame)

    cap.release()

    if not frames_landmarks_list:
        return None

    # Convert list of lists to numpy array (num_frames, num_features)
    landmarks_np = np.array(frames_landmarks_list, dtype=np.float32)

    # Reshape to (num_frames, num_landmarks, 3) as expected by original Preprocessing
    # num_features should be N_LANDMARKS_EXPECTED * 3
    if landmarks_np.shape[1] != N_LANDMARKS_EXPECTED * 3:
        # This might happen if SELECTED_COLUMNS doesn't match N_LANDMARKS_EXPECTED * 3
        # Or if MediaPipe extraction failed for some parts.
        # For now, we'll pad with NaNs if too few, or truncate if too many.
        # A more robust solution would ensure SELECTED_COLUMNS drives this.
        expected_features = N_LANDMARKS_EXPECTED * 3
        if landmarks_np.shape[1] < expected_features:
            padding = np.full((landmarks_np.shape[0], expected_features - landmarks_np.shape[1]), np.nan)
            landmarks_np = np.concatenate([landmarks_np, padding], axis=1)
        else:
            landmarks_np = landmarks_np[:, :expected_features]

    landmarks_np = landmarks_np.reshape(landmarks_np.shape[0], N_LANDMARKS_EXPECTED, 3)

    # --- Apply normalization logic (adapted from ds_2.Preprocessing) ---
    # 1. Normalize: Subtract mean, divide by std dev (calculated on non-NaN values)
    # Create a mask for non-NaN values to compute mean and std
    nonan_mask = ~np.isnan(landmarks_np)

    # Compute mean and std only over valid (non-NaN) data points for each feature column
    # Mean/std should be computed per landmark coordinate (dim 2) across all frames (dim 0) and landmarks (dim 1) that are not NaN

    # Reshape to (total_values_for_coord, 3) to calculate mean/std per x,y,z
    reshaped_for_norm = landmarks_np[nonan_mask].reshape(-1, 3) # This is not quite right
                                                                # Mean/std is channel-wise in original code
                                                                # For each of X, Y, Z, calculate mean/std over all frames and all landmarks

    # Let's do it channel-wise for x, y, z separately
    mean = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)

    for i in range(3): # Iterate over x, y, z
        channel_data = landmarks_np[:, :, i]
        nonan_channel_data = channel_data[~np.isnan(channel_data)]
        if nonan_channel_data.size > 0:
            mean[i] = np.mean(nonan_channel_data)
            std[i] = np.std(nonan_channel_data)
            if std[i] == 0: # Avoid division by zero
                std[i] = 1.0
        else: # All NaNs for this channel
            mean[i] = 0.0
            std[i] = 1.0


    landmarks_np = (landmarks_np - mean[None, None, :]) / std[None, None, :]

    # 2. Fill NaNs: Replace NaNs with 0 (post-normalization)
    landmarks_np[np.isnan(landmarks_np)] = 0.0

    # --- Interpolate or Pad (adapted from ds_2.interpolate_or_pad) ---
    # This needs to be done carefully. The original uses PyTorch's F.interpolate.
    # For simplicity, we'll use OpenCV's resize for downsampling if too long,
    # and pad with zeros if too short.

    current_len = landmarks_np.shape[0]

    if current_len == 0: # Should have been caught earlier
        return None

    processed_frames = np.zeros((MAX_LEN_FRAMES, N_LANDMARKS_EXPECTED, 3), dtype=np.float32)
    input_mask = np.zeros(MAX_LEN_FRAMES, dtype=np.float32)

    if current_len > MAX_LEN_FRAMES:
        # Downsample using interpolation (cv2.resize can work on N-D arrays if channels are last)
        # landmarks_np shape: (current_len, N_LANDMARKS_EXPECTED, 3)
        # We want to resize the time dimension (current_len) to MAX_LEN_FRAMES
        # cv2.resize expects (D, H, W, C) or (H, W, C)
        # We can treat (N_LANDMARKS_EXPECTED, 3) as (width, channels) and current_len as height

        # Reshape for resize: (current_len, N_LANDMARKS_EXPECTED * 3)
        temp_reshaped = landmarks_np.reshape(current_len, -1)
        # Resize: requires (width, height) for dsize
        resized_data = cv2.resize(temp_reshaped, (N_LANDMARKS_EXPECTED * 3, MAX_LEN_FRAMES), interpolation=cv2.INTER_LINEAR)
        # Reshape back: (MAX_LEN_FRAMES, N_LANDMARKS_EXPECTED, 3)
        processed_frames = resized_data.reshape(MAX_LEN_FRAMES, N_LANDMARKS_EXPECTED, 3)
        input_mask[:] = 1.0 # All frames are valid after interpolation
    else:
        # Pad with zeros
        processed_frames[:current_len, :, :] = landmarks_np
        input_mask[:current_len] = 1.0 # Valid frames

    # The model expects input shape (1, MAX_LEN_FRAMES, N_LANDMARKS_EXPECTED, 3)
    # And also an input_mask (1, MAX_LEN_FRAMES)
    # The tflite model converted from cfg_2 might only take the processed_frames as input,
    # and the mask might be implicit or handled by a second input tensor.
    # We need to check input_details of the tflite model.
    # For now, let's assume the first input is the landmark data and the second is the mask.
    # If only one input, it's likely just the landmark data, and masking is internal or not used at inference.

    return processed_frames.astype(input_details[0]['dtype']), input_mask.astype(input_details[1]['dtype'] if len(input_details) > 1 else np.float32)


def postprocess_output(model_output_sequence):
    """Converts model output (sequence of token IDs) to a string."""
    char_list = []
    for token_id in model_output_sequence:
        if token_id == END_TOKEN_ID:
            break
        if token_id == START_TOKEN_ID or token_id == PAD_TOKEN_ID:
            continue
        char_list.append(num_to_character.get(token_id, '?'))
    return "".join(char_list)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return "Error: Model not loaded.", 500

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Save the uploaded file temporarily to pass its path to preprocessing
        # This is necessary because cv2.VideoCapture needs a file path or camera index.
        temp_video_path = "temp_video_file" # Consider adding extension based on file.filename
        try:
            file.save(temp_video_path)

            # Preprocess the video
            # preprocessed_data, input_mask = preprocess_video_frames(temp_video_path) # If two inputs
            processed_input_data, input_mask = preprocess_video_frames(temp_video_path)


            if processed_input_data is None:
                return "Error: Could not process video. No frames found or error in landmark extraction.", 500

            # Prepare input for the TFLite model
            # The model might have one or two inputs (data and mask)
            # Based on `scripts/convert_cfg_2_to_tf_lite.py` and typical model structures,
            # it's likely the TFLite model takes both `frames` and `mask` if `mask` is used by the TF model.
            # Let's check `input_details` again.

            # Assuming input_details[0] is for landmark data, input_details[1] for mask (if present)
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(processed_input_data, axis=0))
            if len(input_details) > 1:
                 # Check if the second input in TFLite model matches the mask's shape
                if input_details[1]['shape'].tolist() == [1, MAX_LEN_FRAMES]:
                    interpreter.set_tensor(input_details[1]['index'], np.expand_dims(input_mask, axis=0))
                else:
                    print(f"Warning: Second TFLite input shape {input_details[1]['shape']} does not match expected mask shape {[1, MAX_LEN_FRAMES]}. Mask might not be used as expected.")

            interpreter.invoke()

            # Output is likely a sequence of token IDs or logits over token IDs.
            # Based on typical sequence-to-sequence models, output_details[0]['index'] would give token IDs directly (after argmax if logits)
            # The tflite model from convert_cfg_2_to_tf_lite.py seems to output generated IDs directly.
            raw_output_sequence = interpreter.get_tensor(output_details[0]['index'])

            # Squeeze batch dimension if present, assuming batch size 1 for inference
            if raw_output_sequence.ndim > 1 and raw_output_sequence.shape[0] == 1:
                model_output_sequence = raw_output_sequence[0]
            else:
                model_output_sequence = raw_output_sequence # if already (seq_len,)

            prediction_text = postprocess_output(model_output_sequence)

        except Exception as e:
            # Log the full error for debugging
            import traceback
            print(f"Error during prediction: {e}\n{traceback.format_exc()}")
            return f"Error during prediction: {e}", 500
        finally:
            # Clean up the temporary file
            import os
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        return render_template('result.html', prediction=prediction_text)

    return "Error processing file", 500

if __name__ == '__main__':
    # Make sure to install mediapipe and opencv-python if not already in requirements
    # pip install flask mediapipe opencv-python tensorflow
    app.run(debug=True, host='0.0.0.0')
