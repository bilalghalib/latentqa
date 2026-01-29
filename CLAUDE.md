# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LatentQA is a research implementation for interpreting and controlling LLM activations using natural language. It trains a "decoder" LLM to:
- **Read**: Answer open-ended questions about a target model's activations in natural language
- **Write/Control**: Steer model behavior by providing gradients from QA-based loss functions

The core method is **Latent Interpretation Tuning (LIT)**: finetuning a decoder LLM on paired datasets of activations and natural language QA pairs, similar to visual instruction tuning.

## Commands

All scripts run from repo root using module notation:

```bash
# Install dependencies
pip install -r requirements.txt

# Training (DDP - multiple GPUs)
torchrun --nnodes 1 --nproc-per-node $NUM_GPUS -m lit.train \
    --target_model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --train_stimulus_completion data/train/stimulus_completion.json \
    --train_stimulus data/train/stimulus.json \
    --train_control data/train/control.json \
    --train_qa data/train/qa.json \
    --gradient_accumulation_steps 8

# Training (FSDP - for 70B models, tested on 8x A100-80GB)
torchrun --nnodes 1 --nproc-per-node 8 -m lit.train \
    --use_fsdp --min_layer_to_read 21 --max_layer_read 22

# Reading activations (interpret model behavior)
python3 -m lit.reading \
    --target_model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --decoder_model_name aypan17/latentqa_llama-3-8b-instruct

# Control/Steering (modify model behavior)
python3 -m lit.control \
    --decoder_model_name aypan17/latentqa_llama-3-8b-instruct \
    --control promote_veganism \
    --dataset dolly \
    --samples 30 \
    --per_layer_loss
```

## Architecture

### Core Pipeline

```
Target LLM → Activation Hooking (layer k=15) → Decoder LLM (write to layer ℓ=0) → QA Output
                      ↓
              [Activations patched into decoder embeddings]
```

**Key insight**: Read from middle layers (k=15) which have semantically-rich representations; write to layer 0 to give decoder maximum processing steps.

### Main Components

- **`lit/train.py`**: Multi-GPU training with DDP/FSDP, LoRA finetuning, EMA weight tracking
- **`lit/reading.py`**: `INTERPRET([Act], question)` - decode activations to natural language
- **`lit/control.py`**: `STEER([Act], control)` - backprop QA loss to modify target model

### Activation Utilities (`lit/utils/activation_utils.py`)

- `_forward_cache_outputs()`: Hook to capture layer activations
- `generate_substitute_layer_single()`: Replace decoder embeddings with target activations
- `latent_qa()`: Main pipeline orchestrating read/write operations

### Data Types (three training data types)

1. **control**: Activations from control prompt only
2. **stimulus**: Activations from stimulus with control masked
3. **stimulus+completion**: Activations from full dialog with control masked

### Supported Models

Llama-3 (8B, 70B), Llama-3.1, Mistral, DeepSeek, Gemma-3, Qwen-3

## Key Configuration

- **Read layer**: k=15 (middle layers have richest semantic content)
- **Write layer**: ℓ=0 (maximize decoder processing)
- **LoRA**: rank 32, alpha 64 on attention + MLP modules
- **Learning rate**: 10^-4, batch size 128
- **Training**: 4x A100s minimum

Config files: `lit/configs/train_config.py`, `lit/configs/steer_config.py`, `lit/configs/interpret_config.py`

## Data Format

Training data is JSON with triples: `(control+stimulus, completion, QA pairs)`

Example:
```json
{
  "control": "Be a pirate",
  "stimulus": "What color is the sky?",
  "completion": "It be blue, matey!",
  "qa": [["How will the model speak?", "Like a pirate"]]
}
```

## Output Directories

- `out/runs/`: Training checkpoints
- `out/completions/`: Inference results
- `controls/`: QA-pair JSON files for steering (e.g., `golden_gate.json`, `harry_potter.json`)

## Pre-trained Decoder

Available at: `aypan17/latentqa_llama-3-8b-instruct` (HuggingFace)
