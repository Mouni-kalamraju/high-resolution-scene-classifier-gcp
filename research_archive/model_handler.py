import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2

# --- Configuration ---
IMG_SIZE = (150, 150)
# Make sure this points to your .weights.h5 or .keras file
MODEL_PATH = 'intel_classifier_weights.weights.h5' 
CLASSES_PATH = 'class_names.txt'

model = None
class_names = []

def build_model_manually():
    """Explicitly defines the architecture to avoid the 2-input error."""
    inputs = Input(shape=(150, 150, 3))
    
    # Load architecture only
    base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    # Explicitly connect the layers
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)
    
    new_model = Model(inputs, outputs)
    return new_model

def load_assets():
    global model
    global class_names
    
    print("Attempting to build model and load weights...")
    try:
        # 1. Manually build the architecture
        model = build_model_manually()
        
        # 2. Load ONLY the weights into that architecture
        # This prevents Keras from trying to 'reconstruct' the broken graph
        model.load_weights(MODEL_PATH)
        print("Model weights loaded successfully!")
        
        # 3. Load classes
        with open(CLASSES_PATH, 'r') as f:
            class_names = [line.strip() for line in f]
            
    except Exception as e:
        print(f"FAILED TO LOAD: {e}")
        raise e

# Call this so it loads when the server starts
load_assets()

def preprocess_image(image_bytes):
    """Converts image bytes to a normalized NumPy array."""
    try:
        # Open image from bytes
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Resize to the required input size
        image = image.resize(IMG_SIZE)
        # Convert to NumPy array
        image_array = np.array(image)
        # Add batch dimension (1, 150, 150, 3)
        image_array = np.expand_dims(image_array, axis=0)
        # Normalize (must match the 1./255 preprocessing in Colab)
        image_array = image_array / 255.0
        
        return image_array
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def predict_image(image_array):
    """Generates prediction and returns human-readable result."""
    if image_array is None:
        return "Error", 0.0
        
    # Generate prediction (returns probabilities for 6 classes)
    predictions = model.predict(image_array)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions[0])
    
    # Get the class name and confidence score
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])
    
    return predicted_class, confidence

