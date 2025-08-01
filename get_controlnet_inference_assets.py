import os
import urllib.request
import sys
# --- Paths ---
work_dir = os.path.dirname(os.path.abspath(__file__))
controlnet_dir = os.path.join(work_dir, "ControlNet")
models_dir = os.path.join(controlnet_dir, "models")

# --- Create directories if missing ---
os.makedirs(models_dir, exist_ok=True)

print(models_dir, os.path.isdir(models_dir))
# --- Files to download ---
files_to_download = [
    {
        "url": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth",
        "dest": os.path.join(models_dir, "control_sd15_canny.pth"),
    },
    {
        "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt",
        "dest": os.path.join(models_dir, "v1-5-pruned.ckpt"),
    },
    {
        "url": "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v15.yaml",
        "dest": os.path.join(models_dir, "cldm_v15.yaml"),
    }
]
def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
    bar = "#" * (percent // 2) + "-" * (50 - percent // 2)
    sys.stdout.write(f"\r  [{bar}] {percent}%")
    sys.stdout.flush()
    if downloaded >= total_size:
        sys.stdout.write("\n")

# --- Download logic ---
def download_file(url, dest):
    try:
        # Check remote file size
        with urllib.request.urlopen(url) as response:
            remote_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and compare size
        if os.path.exists(dest):
            local_size = os.path.getsize(dest)
            if local_size == remote_size:
                print(f"✅ Already exists and up to date: {os.path.basename(dest)}")
                return
            else:
                print(f"⚠️ File size mismatch. Re-downloading: {os.path.basename(dest)}")

        # Download file
        print(f"⬇ Downloading: {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest, reporthook=progress_bar)
        print(f"✅ Downloaded: {os.path.basename(dest)}")

    except Exception as e:
        print(f"❌ Failed to download {url}:\n{e}")

# --- Run downloads ---
if __name__ == "__main__":
    print("📦 Downloading ControlNet inference requirements...")
    for item in files_to_download:
        download_file(item["url"], item["dest"])
    print("\n✅ All downloads complete.")
