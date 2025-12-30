from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import model_handler # Import our logic file

# Initialize FastAPI application
app = FastAPI(
    title="Intel Scene Classifier API",
    description="Real-time image classification service for 6 scene categories."
)

# Pydantic model for a clean, structured response
class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    message: str

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the Intel Scene Classifier API. Go to /docs for the API interface."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the predicted scene category.
    """
    # 1. Read the image file contents
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")
        
    # 2. Preprocess the image using the handler
    processed_array = model_handler.preprocess_image(image_bytes)
    
    if processed_array is None:
        raise HTTPException(status_code=422, detail="Invalid image format or preprocessing failed.")

    # 3. Generate prediction
    predicted_class, confidence = model_handler.predict_image(processed_array)

    # 4. Return the structured response
    return PredictionResponse(
        filename=file.filename,
        predicted_class=predicted_class,
        confidence=round(confidence, 4), # Round confidence for clean output
        message="Classification successful."
    )