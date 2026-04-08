#!/usr/bin/env python3
"""
Utility functions for modular analyses.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from transformer_lens import HookedTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoImageProcessor, AutoModel, ResNetForImageClassification
from typing import List, Dict, Union
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path

import sys
sys.path.append('./train')
from model import ResNetMLP, ResNetMLPSkeleton

# Dictionary of no-LayerNorm models with standardized names
NO_LAYERNORM_MODELS = {
    "gpt2-small_LNFree": "schaeff/gpt2-small_LNFree300",
    "gpt2-medium_LNFree": "schaeff/gpt2-medium_LNFree500", 
    "gpt2-large_LNFree": "schaeff/gpt2-large_LNFree600",
    "gpt2-xl_LNFree": "schaeff/gpt2-xl_LNFree800"
}

LAYER_ARGUMENT_IDX_MAPPING = {
    'toy_resnet': {
        'logits': 'logits',
        'input': -2,
        'embed': -1
    },
}

def load_gpt2_regular(model_name):
    """Load regular GPT-2 with LayerNorm."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Map model names to actual model names
    model_mapping = {
        "gpt2-small": "gpt2",
        "gpt2-medium": "gpt2-medium", 
        "gpt2-large": "gpt2-large",
        "gpt2-xl": "gpt2-xl"
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown regular model: {model_name}. Available: {list(model_mapping.keys())}")
    
    actual_model_name = model_mapping[model_name]
    model = HookedTransformer.from_pretrained(actual_model_name, fold_ln=False, center_unembed=False).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(actual_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device, model_name

def load_gpt2_no_ln(model_name):
    """Load GPT-2 without LayerNorm from the specified model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_name not in NO_LAYERNORM_MODELS:
        raise ValueError(f"Unknown no-LayerNorm model: {model_name}. Available: {list(NO_LAYERNORM_MODELS.keys())}")
    
    hf_model_path = NO_LAYERNORM_MODELS[model_name]
    
    # Load the HuggingFace model
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_path).to("cpu")
    
    # Undo hacky LayerNorm removal
    for block in hf_model.transformer.h:
        block.ln_1.weight.data = block.ln_1.weight.data / 1e6
        block.ln_1.eps = 1e-5
        block.ln_2.weight.data = block.ln_2.weight.data / 1e6
        block.ln_2.eps = 1e-5
    hf_model.transformer.ln_f.weight.data = hf_model.transformer.ln_f.weight.data / 1e6
    hf_model.transformer.ln_f.eps = 1e-5
    
    # Properly replace LayerNorms by Identities
    def removeLN(transformer_lens_model):
        for i in range(len(transformer_lens_model.blocks)):
            transformer_lens_model.blocks[i].ln1 = torch.nn.Identity()
            transformer_lens_model.blocks[i].ln2 = torch.nn.Identity()
        transformer_lens_model.ln_final = torch.nn.Identity()
    
    # Determine the base model size for TransformerLens
    if "small" in model_name:
        base_model = "gpt2"
    elif "medium" in model_name:
        base_model = "gpt2-medium"
    elif "large" in model_name:
        base_model = "gpt2-large"
    elif "xl" in model_name:
        base_model = "gpt2-xl"
    
    model = HookedTransformer.from_pretrained(base_model, hf_model=hf_model, fold_ln=True, center_unembed=False).to(device)
    removeLN(model)
    model.cfg.normalization_type = None
    
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device, model_name

def load_vit_regular(model_name):
    """Load regular ViT with LayerNorm."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    actual_model_name = model_name
    model = AutoModel.from_pretrained(actual_model_name).to(device)
    processor = AutoImageProcessor.from_pretrained(actual_model_name)

    return model, processor, device, model_name

def load_resnet(model_name):
    """Load ResNet."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    actual_model_name = model_name
    model = ResNetForImageClassification.from_pretrained(actual_model_name).to(device)
    processor = AutoImageProcessor.from_pretrained(actual_model_name)

    return model, processor, device, model_name

def load_model(model_name):
    """Load either regular, no-LayerNorm GPT-2, or ViT model."""
    if 'LNFree' in model_name:
        model, tokenizer, device, actual_name = load_gpt2_no_ln(model_name)
    elif "gpt2" in model_name:
        model, tokenizer, device, actual_name = load_gpt2_regular(model_name)
    elif "vit" in model_name or "dino" in model_name:
        model, processor, device, actual_name = load_vit_regular(model_name)
    elif "resnet" in model_name:
        model, processor, device, actual_name = load_resnet(model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Store the original model name for saving
    model.original_model_name = model_name
    return model

def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_config = checkpoint['config']
    model_config = full_config['model']
    model_type = full_config['model_type']

    if model_type == "ResNetMLP":
        model = ResNetMLP(**model_config)
    elif model_type == "ResNetMLPSkeleton":
        model = ResNetMLPSkeleton(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, model_config, full_config

def get_n_layers_from_model(model):
    """Get number of layers from a model, handling both HookedTransformer and ViT models."""
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'n_layers'):
        # HookedTransformer model
        return model.cfg.n_layers
    elif hasattr(model, 'vit') and hasattr(model.vit, 'encoder') and hasattr(model.vit.encoder, 'layer'):
        # ViT model
        return len(model.vit.encoder.layer)
    elif hasattr(model, 'resnet'):
        # ResNetForImageClassification
        return len(model.resnet.encoder.stages)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'stages'):
        # ResNetModel
        return len(model.encoder.stages)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # Some other transformer models
        return len(model.encoder.layer)
    else:
        print(model.__class__)
        raise ValueError(f"Cannot determine number of layers for model type: {type(model)}")


def construct_filepath(model_name: str, shared_id: str, interpolation_layer: int, pair_ids: List[str], n_steps: int, freeze_suffix: str = "") -> str:
    """Construct full filepath for activation file.

    Args:
        freeze_suffix: Optional freeze suffix like "_freeze_attn" or "_freeze_mlp"
    """
    context_clean = shared_id.replace(" ", "_").replace(".", "").replace(",", "").replace("'", "").replace('"', "")
    tokens_str = "_".join(pair_ids)
    filename = f"interpolate_layer{interpolation_layer}{freeze_suffix}_{context_clean}_[{tokens_str}]_{n_steps}steps.pt"
    return f"./activations/{model_name}/{filename}"


def load_config(config_path: str = "./vis_plots/config.yaml") -> Dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_name(config: Dict, model_type: str) -> str:
    """Get a single model name/path from config. Returns first element if list, string if string."""
    val = config['model_names'][model_type]
    if isinstance(val, list):
        return val[0]
    return val

def _find_latest_checkpoint(directory: str) -> str:
    """Find the latest checkpoint file in a directory."""

    # Return None as fallback if the given path is not a directory
    if not os.path.isdir(directory):
        print(f"Warning: {directory} is not a directory. Skipping.")
        return None

    checkpoint_files = [f for f in os.listdir(directory) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {directory}")
    
    # Extract epoch numbers and find the latest
    def extract_epoch(filename):
        try:
            return int(filename.split("checkpoint_epoch_")[1].split(".pt")[0])
        except:
            return -1  # In case of unexpected filename format

    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    return os.path.join(directory, latest_checkpoint)

def get_model_names(config: Dict, model_type: str) -> List[str]:
    """Get all model names/paths from config as a list. Always returns a list."""
    val = config['model_names'][model_type]
    if isinstance(val, list):
        return val
    elif isinstance(val, str):
        path = Path(val)
        if path.is_file() and path.suffix == ".pt":
            return [val]
        else: # when given a directory, return all .pt files in that directory
            latest_ckpts = [_find_latest_checkpoint(os.path.join(val, timestamp)) for timestamp in os.listdir(val)]
            return [ckpt for ckpt in latest_ckpts if ckpt is not None]
    else:
        raise ValueError(f"Invalid model name format for {model_type}: {val}")


def get_n_layers(activations: Dict) -> int:
    """Get number of layers from activation dictionary keys."""
    max_layer = max([int(k.split('_')[0].replace('layer', ''))
                     for k in activations.keys() if k.startswith('layer')])
    return max_layer + 1
