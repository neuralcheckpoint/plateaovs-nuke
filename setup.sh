#!/usr/bin/env bash
# Build the PlateAOVs engine: GPU venv, Cosmos DiffusionRenderer source and
# the inverse renderer's weights (27 GB) with its video tokenizer (1.8 GB),
# kept in models/checkpoints. No Transformer Engine: plateaovs/model.py
# stands in for it with PyTorch. Safe to re-run; each step skips what is
# already in place.
set -euo pipefail
cd "$(dirname "$0")"

COSMOS_COMMIT=0f3e2dc

if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.10 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt

if [ ! -d vendor/cosmos1-diffusion-renderer ]; then
    git clone https://github.com/nv-tlabs/cosmos1-diffusion-renderer.git vendor/cosmos1-diffusion-renderer
    git -C vendor/cosmos1-diffusion-renderer checkout --quiet "$COSMOS_COMMIT"
fi

for repo in nvidia/Cosmos-Tokenize1-CV8x8x8-720p nvidia/Diffusion_Renderer_Inverse_Cosmos_7B; do
    .venv/bin/hf download "$repo" --local-dir "models/checkpoints/${repo#*/}"
done

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "PlateAOVs engine ready."
