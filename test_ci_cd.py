import base64
import os
from google.cloud import aiplatform

# --- CONFIGURATION (via Environment Variables) ---

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id-placeholder")
ENDPOINT_ID = os.getenv("GCP_ENDPOINT_ID", "your-endpoint-id-placeholder")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

TEST_IMAGE_PATH = os.getenv("TEST_IMAGE_PATH", "test_image.jpg")

def test_prediction():
    
    if "placeholder" in PROJECT_ID or "placeholder" in ENDPOINT_ID:
        print(" Warning: Project/Endpoint IDs not set. Use environment variables.")
        return

    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f" Error: Image file not found at {TEST_IMAGE_PATH}")
        return

    with open(TEST_IMAGE_PATH, "rb") as f:
        file_content = f.read()
        encoded_content = base64.b64encode(file_content).decode("utf-8")


    instance = {"content": encoded_content} 

    print(f"Sending prediction request to {ENDPOINT_ID}...")
    
    try:
        # Standard Vertex AI prediction call
        response = endpoint.predict(instances=[instance])
        
        print("\n Prediction Successful!")
        print(f"Result: {response.predictions}")
            
    except Exception as e:
        print(f"\n Prediction Failed: {e}")

if __name__ == "__main__":
    test_prediction()