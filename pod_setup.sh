#!/bin/bash

# 1. Create and activate environment
conda create -n unite python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate unite

# 2. Install Core ML Stack
# Note: Using pip install for torch is fine, but ensure it matches your CUDA version if needed
pip install torch torchvision torchaudio

# 3. Install Transformers and Accelerate
pip install transformers accelerate

# 4. Install fixed versions for Python 3.9 stability
# datasets 2.18.0 is a very stable release for Python 3.9
# huggingface-hub 0.20.0+ provides full login/API support without the typing bugs
pip install "aiohttp<3.9.0" "datasets>=2.16.0,<3.0.0" "huggingface_hub>=0.20.0"

# 5. Clean up typing-extensions (to ensure no unhashable list errors)
pip install "typing-extensions>=4.10.0"

echo "Setup complete. To log in to Hugging Face, run: huggingface-cli login"