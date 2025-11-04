import argparse
import logging
from io import BytesIO
from typing import Dict
import sys
import torch
import mlflow.pytorch
from fastapi import FastAPI, Request
from PIL import Image
from torchvision import transforms
import uvicorn

# Setup logging to stdout for controller visibility
logger = logging.getLogger("mnist_serve")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Script is started...")  # initial log

# Init FastAPI
app = FastAPI()

# Globals
model = None
device = None
preprocessor = None
model_loaded = False


@app.on_event("startup")
async def load_model():
    global model, device, preprocessor, model_loaded

    logger.info("Starting model loading...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        model = mlflow.pytorch.load_model(app.state.model_path, map_location=device)
        model.to(device)
        model.eval()
        model_loaded = True
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False

    preprocessor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    logger.info("Preprocessor setup done.")


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    status = "ok" if model_loaded else "error"
    logger.info(f"Health check called, status: {status}")
    return {"status": status}


@app.post("/predict")
async def predict(request: Request) -> Dict:
    global model, device, preprocessor

    if not model_loaded:
        logger.warning("Predict called but model not loaded!")
        return {"error": "Model not loaded"}

    # Read image
    image_payload_bytes = await request.body()
    pil_image = Image.open(BytesIO(image_payload_bytes))
    logger.info(f"[1/3] Parsed image data: {pil_image}")

    # Preprocess
    input_tensor = preprocessor(pil_image).unsqueeze(0).to(device)
    logger.info(f"[2/3] Image transformed, tensor shape: {input_tensor.shape}")

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    logger.info("[3/3] Inference done!")

    class_idx = int(torch.argmax(output_tensor[0]))
    logger.info(f"Predicted class index: {class_idx}")
    return {"class_index": class_idx}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to MLflow PyTorch model")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Save parsed args in app.state for startup
    app.state.model_path = args.model_path
    app.state.host = args.host
    app.state.port = args.port

    logger.info(f"Starting uvicorn server at {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

