import torch
import importlib
import sys
import subprocess
import argparse
import os

# Define Grad-TTS project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, 'Speech-Backbones', 'Grad-TTS')
sys.path.insert(0, project_path)

# Import params module
params = importlib.import_module("params")

# Parse arguments
parser = argparse.ArgumentParser(description="Grad-TTS Dynamic Inference Launcher")
parser.add_argument("--model", type=str, required=True, choices=["grad-tts", "grad-tts-old", "grad-tts-libri-tts"],
                    help="Which pretrained model to use.")
parser.add_argument("--timesteps", type=int, default=5, help="Number of timesteps (t)")
parser.add_argument("--speaker", type=int, default=0, help="Speaker ID to synthesize")
args = parser.parse_args()

# Resolve checkpoint path
ckpt_filename = f"{args.model}.pt"
ckpt_path = os.path.join(project_path, "checkpts", ckpt_filename)

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu')

# Automatically determine n_spks:
if args.model == "grad-tts-libri-tts":
    # LibriTTS pretrained model has 247 speakers
    n_spks = 247
elif args.model == "grad-tts":
    # Your custom model (adjust accordingly if needed)
    n_spks = 1
elif args.model == "grad-tts-old":
    # Example: let's say old model was also single speaker
    n_spks = 1
else:
    n_spks = 1

# Override params dynamically
params.n_spks = n_spks

print(f"Running inference with model: {args.model}")
print(f"n_spks automatically set to: {n_spks}")
print(f"Timesteps: {args.timesteps}")
print(f"Speaker ID: {args.speaker}")

# Build inference command
inference_command = [
    "python",
    "inference.py",
    "-f", os.path.join(project_path, "resources", "filelists", "synthesis.txt"),
    "-c", ckpt_path,
    "-t", str(args.timesteps),
]

# Only add speaker argument if multi-speaker model
if n_spks > 1:
    inference_command += ["-s", str(args.speaker)]

if n_spks == 1 and args.speaker != 0:
    print("Warning: Single speaker model — ignoring provided speaker ID.")


# Run inference inside Grad-TTS directory
subprocess.run(inference_command, cwd=project_path)
