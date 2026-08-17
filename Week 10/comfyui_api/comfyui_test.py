import requests
import time

COMFY_URL = "http://127.0.0.1:8188"

# Test connection
response = requests.get(f"{COMFY_URL}/system_stats")

print("ComfyUI status:", response.status_code)

if response.status_code == 200:
    print("✅ Python successfully connected to ComfyUI!")
else:
    print("❌ Could not connect to ComfyUI")