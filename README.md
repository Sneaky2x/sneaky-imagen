# Sneaky Imagen Serverless Worker

This worker is a custom serverless endpoint for RunPod designed for high-performance image generation.

## Directory Structure

- `Dockerfile`: Defines the environment and installation steps.
- `builder.py`: Python script that runs *during the Docker build* to download models from CivitAI/HuggingFace.
- `handler.py`: The entry point for the RunPod worker. It loads the models and handles inference requests.
- `requirements.txt`: Python dependencies.

## Setup

1. **Configure Models**:
   Edit `builder.py` to change the `CHECKPOINT_URL`, `LORA_URLS`, or `VAE_URL` if you want to bake different models into the image.
   
   *Current defaults:*
   - Checkpoint: CivitAI Model ID 2255476 (SDXL)
   - LoRA: CivitAI Model ID 1326524
   - VAE: CivitAI Model ID 333245

2. **Build the Image**:
   You need to build this Docker image and push it to a registry (Docker Hub, GHCR, etc.) that RunPod can access.

   ```bash
   docker build -t your-username/runpod-sdxl-worker:v1 .
   docker push your-username/runpod-sdxl-worker:v1
   ```

3. **Deploy on RunPod**:
   - Go to RunPod > Templates > New Template.
   - **Image Name**: `your-username/runpod-sdxl-worker:v1`
   - **Container Disk**: 10GB+ (Models are large).
   - **Env Variables**:
     - `MODEL_TYPE`: `SDXL` (or `Flux` if you changed the model).
   - Create a Serverless Endpoint using this template.

## API Usage

**Input Payload:**
```json
{
  "input": {
    "prompt": "a beautiful landscape, 8k",
    "negative_prompt": "blurry, ugly",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "cfg_scale": 5.5,
    "seed": 123456,
    "scheduler": "Euler a"
  }
}
```

**Response:**
```json
{
  "image": "<base64_encoded_string>",
  "seed": 123456,
  "params": { ... }
}
```
