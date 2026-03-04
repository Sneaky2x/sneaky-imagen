import os
import requests
import re
from pathlib import Path

# Configuration for Sneaky Imagen
# You can customize these URLs before building the image
CHECKPOINT_URL = "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor&size=full&fp=fp16"
LORA_URLS = ["https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor"]
VAE_URL = "https://civitai.com/api/download/models/333245?type=Model&format=SafeTensor"

# Optional: Set a CivitAI token if downloading restricted models
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "daa65fe2bceb540c0a1a9e7cf2ab1245") 

MODELS_DIR = "/models"
CHECKPOINT_DIR = f"{MODELS_DIR}/checkpoints"
LORA_DIR = f"{MODELS_DIR}/loras"
VAE_DIR = f"{MODELS_DIR}/vae"

def download_file(url, output_dir, token=None):
    os.makedirs(output_dir, exist_ok=True)
    
    headers = {}
    if 'civitai.com' in url and token:
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}token={token}"

    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
        response.raise_for_status()

        # Try to get filename from content-disposition
        content_disp = response.headers.get('content-disposition', '')
        filename_match = re.findall(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disp)
        
        if filename_match and filename_match[0][0]:
            filename = filename_match[0][0].strip('\'"')
        else:
            filename = "model.safetensors" # Fallback
            
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            print(f"File {filename} already exists, skipping.")
            return output_path

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024*1024*100) == 0: # Print every 100MB
                        print(f"Downloaded {downloaded/1024/1024:.2f} MB")
                        
        print(f"Downloaded {filename} to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

if __name__ == "__main__":
    print("Starting build-time model download...")
    
    # Download Checkpoint
    if CHECKPOINT_URL:
        download_file(CHECKPOINT_URL, CHECKPOINT_DIR, CIVITAI_TOKEN)
        
    # Download LoRAs
    for url in LORA_URLS:
        if url:
            download_file(url, LORA_DIR, CIVITAI_TOKEN)
            
    # Download VAE
    if VAE_URL:
        download_file(VAE_URL, VAE_DIR, CIVITAI_TOKEN)
        
    print("Build-time download complete.")
