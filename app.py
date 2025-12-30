import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from PIL import Image
import io
import base64
from contextlib import asynccontextmanager

# Dictionary to hold the model state
ml_models = {}

# --- 1. LIFESPAN LOGIC (Loads model on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📢 DEBUG: Starting lifespan...", flush=True)
    try:
        print("📢 DEBUG: Loading .keras file now...", flush=True)
        # Load model logic
        ml_models["classifier"] = tf.keras.models.load_model("model_functional_final.keras")
        print("✅ DEBUG: Model Load COMPLETE!", flush=True)
    except Exception as e:
        print(f"❌ DEBUG: ERROR LOADING MODEL: {e}", flush=True)
    yield
    # --- SHUTDOWN ---
    print("🧹 Lifespan: Shutting down, clearing model...")
    ml_models.clear()

# --- 2. CONNECT LIFESPAN ---
app = FastAPI(
    title="Intel Scene Classifier: Fast Ensemble",
    version="2.0.0",
    lifespan=lifespan  
)

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- 3. HEALTH CHECK  ---
@app.get("/health", status_code=200)  
async def health_check():
    is_loaded = "classifier" in ml_models
    if not is_loaded:
        # Return 503 so Vertex knows we aren't ready yet
        return {"status": "STARTING", "model_loaded": False}, 503
    return {"status": "OK", "model_loaded": True}

def preprocess_method_5(image_bytes):
    # 1. Convert bytes to RGB
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_rgb = np.array(img)

    # STREAM 1: Baseline
    stream_a = cv2.resize(img_rgb, (150, 150), interpolation=cv2.INTER_AREA)
    stream_a = np.expand_dims(stream_a.astype('float32'), axis=0)

    # STREAM 2: Smart (Local Features)
    mid_res = cv2.resize(img_rgb, (300, 300), interpolation=cv2.INTER_AREA)
    smooth = cv2.bilateralFilter(mid_res, d=5, sigmaColor=50, sigmaSpace=50)
    
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    stream_b = cv2.resize(enhanced, (150, 150), interpolation=cv2.INTER_AREA)
    stream_b = np.expand_dims(stream_b.astype('float32'), axis=0)

    return stream_a, stream_b

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None)):
    if "classifier" not in ml_models:
        return {"error": "Model is not initialized yet"}
    
    # --- 4. HANDLE VERTEX AI JSON FORMAT ---
    try:
        if file:
            # Handle local curl test
            contents = await file.read()
        else:
            # Handle Vertex AI Cloud Request
            body = await request.json()
            b64_string = body['instances'][0]['content']
            contents = base64.b64decode(b64_string)

        input_a, input_b = preprocess_method_5(contents)
        
        model = ml_models["classifier"]
        pred_a = model.predict(input_a, verbose=0)
        pred_b = model.predict(input_b, verbose=0)

        final_probs = (pred_a + pred_b) / 2
        idx = np.argmax(final_probs)
        confidence = float(np.max(final_probs))

        return {
            "predictions": [
                {
                    "label": CLASS_NAMES[idx],
                    "confidence": f"{confidence:.2%}",
                    "method": "Fast Ensemble (Method 5)"
                }
            ]
        }
    except Exception as e:
        return {"error": str(e)}