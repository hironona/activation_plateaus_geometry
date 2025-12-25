#!/usr/bin/env python3
"""
Interpolate activations between token pairs in transformer models.

For each token pair and each layer:
1. Collect resid_post activations for both tokens at all layers
2. Interpolate between them at the specified layer using SLERP
3. Inject interpolated activations and propagate through subsequent layers
4. Record activations (attn_out, resid_mid, mlp_post, mlp_out, resid_post) and logits

Outputs: One file per (token_pair, interpolation_layer, freeze_mode) combination
"""

import torch
import os
import argparse
from typing import Dict
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import sys
sys.path.append('./scripts')
from utils import load_model, load_config, slerp_rescale, construct_filepath, get_n_layers_from_model

config = load_config()
N_STEPS = config['n_steps']

# Hook types to record at each layer
ACTIVATION_HOOKS = [
    ('attn_out', 'blocks.{}.hook_attn_out'),
    ('resid_mid', 'blocks.{}.hook_resid_mid'),
    ('mlp_post', 'blocks.{}.mlp.hook_post'),
    ('mlp_out', 'blocks.{}.hook_mlp_out'),
    ('resid_post', 'blocks.{}.hook_resid_post'),
]

#TODO: create separate hooking functions, both for collecting and interpolating, for different model types

def collect_original_activations_hooked_transformer(model: HookedTransformer, shared_context: str, variable_data: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Collect resid_post activations at all layers for a given token.

    Returns dict with keys like 'layer{i}_resid_post' -> tensor of shape [1, hidden_dim]
    """
    full_sequence = f"{shared_context} {variable_data}"
    tokens = model.to_tokens(full_sequence, prepend_bos=False)

    activations = {}

    def create_collection_hook(layer_idx):
        def hook_fn(activation, hook):
            activations[f'layer{layer_idx}_resid_post'] = activation[:, -1, :].cpu().clone()
            return activation
        return hook_fn

    hooks = [(f'blocks.{i}.hook_resid_post', create_collection_hook(i))
             for i in range(model.cfg.n_layers)]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")

    return activations

def collect_original_activations_vit(model: ViTForImageClassification, processor: ViTImageProcessor, variable_data: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Collect resid_post activations at all layers for the CLS token of the given image.

    Returns dict with keys like 'layer{i}_resid_post' -> tensor of shape [1, hidden_dim]
    """

    image = Image.open(requests.get(variable_data, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt").to(device)

    activations = {}

    def create_collection_hook(layer_idx):
        """Create hook to collect CLS token activation at each layer."""
        def hook_fn(module, input, output):
            # For ViT, output is typically (batch_size, num_patches + 1, hidden_dim)
            # We take the CLS token (first token) activation
            if isinstance(output, tuple):
                output = output[0]
            # output shape: [batch_size, seq_len, hidden_dim]
            # CLS token is at index 0
            activations[f'layer{layer_idx}_resid_post'] = output[:, 0, :].cpu().clone()
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.vit.encoder.layer):
        hook = layer.register_forward_hook(create_collection_hook(i))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Clean up
    del inputs
    torch.cuda.empty_cache() if device == 'cuda' else None

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations

def collect_original_activations_wrapper(model_type: str, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Wrapper to collect original activations from different model types.
    
    :param model_type: Type of model ('hooked_transformer' or 'vit')
    :type model_type: str
    :param kwargs: Model-specific arguments (model, processor, shared_context, variable_data, device)
    :return: Dictionary mapping layer names to activation tensors
    :rtype: Dict[str, torch.Tensor]
    :raises ValueError: If model_type is not supported
    """
    if model_type not in ['hooked_transformer', 'vit']:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if model_type == 'hooked_transformer':
        return collect_original_activations_hooked_transformer(**kwargs)
    
    if model_type == 'vit':
        return collect_original_activations_vit(**kwargs)
    
def interpolate_resid_post_layer(
    model: HookedTransformer,
    shared_context: str,
    resid_post_a: Dict[str, torch.Tensor],
    resid_post_b: Dict[str, torch.Tensor],
    interpolation_layer: int,
    n_steps: int,
    device: str,
    freeze_attention: bool = False,
    freeze_mlp: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Interpolate resid_post at specified layer and record all downstream activations.

    Returns dict with keys like 'layer{i}_{hook_name}' -> tensors of shape [n_steps, seq_len, hidden_dim]
    and 'logits' -> tensor of shape [n_steps, seq_len, vocab_size]
    """

    # Get resid_post activations at the interpolation layer for both tokens
    resid_a = resid_post_a[f'layer{interpolation_layer}_resid_post']  # [1, hidden_dim]
    resid_b = resid_post_b[f'layer{interpolation_layer}_resid_post']  # [1, hidden_dim]

    # Compute SLERP interpolations on device
    resid_a_device = resid_a.to(device)
    resid_b_device = resid_b.to(device)
    alphas = torch.linspace(0, 1, n_steps, device=device)

    interpolated_resid_post = torch.stack([
        slerp_rescale(resid_a_device, resid_b_device, alpha.item()).squeeze(0)
        for alpha in alphas
    ])  # [n_steps, hidden_dim] on device

    # Free up device memory immediately
    del resid_a_device, resid_b_device, alphas

    # Storage for collected activations (keep interpolated values on device for injection)
    activations = {f'layer{interpolation_layer}_resid_post': interpolated_resid_post.cpu().clone()}

    # Hook to inject interpolated activations (already on device, no transfer needed)
    def inject_hook(activation, hook):
        # interpolated_resid_post is [n_steps, hidden_dim], activation is [n_steps, seq_len, hidden_dim]
        activation[:, -1, :] = interpolated_resid_post
        return activation

    # Hook to collect and optionally freeze activations
    def create_collection_hook(hook_name, target_layer):
        def hook_fn(activation, hook):
            # Freeze to mean across all steps if requested
            if freeze_attention and hook_name == 'attn_out':
                mean_activation = activation[:, -1, :].mean(dim=0, keepdim=True)
                activation[:, -1, :] = mean_activation.expand(activation.shape[0], -1)
            elif freeze_mlp and hook_name == 'mlp_out':
                mean_activation = activation[:, -1, :].mean(dim=0, keepdim=True)
                activation[:, -1, :] = mean_activation.expand(activation.shape[0], -1)
            activations[f'layer{target_layer}_{hook_name}'] = activation.cpu().clone()
            return activation
        return hook_fn

    # Build hooks list
    hooks = [(f'blocks.{interpolation_layer}.hook_resid_post', inject_hook)]

    for target_layer in range(interpolation_layer, model.cfg.n_layers):
        for hook_name, hook_pattern in ACTIVATION_HOOKS:
            hooks.append((hook_pattern.format(target_layer), create_collection_hook(hook_name, target_layer)))

    # Create input tokens (context + dummy token that gets replaced)
    context_tokens = model.to_tokens(shared_context, prepend_bos=False)
    full_tokens = torch.cat([context_tokens[0], torch.tensor([0], device=device)])
    batched_tokens = full_tokens.unsqueeze(0).repeat(n_steps, 1)

    # Run forward pass with all interpolated activations in parallel
    with torch.no_grad():
        logits = model.run_with_hooks(batched_tokens, fwd_hooks=hooks, return_type="logits")

    activations['logits'] = logits.cpu().clone()

    # Clean up device memory
    del interpolated_resid_post, logits

    return activations

def interpolate_vit_layer(
    model,
    processor,
    shared_images: Dict[str, str], # urls_to_images
    resid_post_a: Dict[str, torch.Tensor],
    resid_post_b: Dict[str, torch.Tensor],
    interpolation_layer: int,
    n_steps: int,
    device: str,
    freeze_attention: bool = False,
    freeze_mlp: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Interpolate at specified layer in ViT and record all downstream activations.
    Interpolates only the CLS token activation.
    
    Args:
        model: ViT model
        processor: ViT processor
        resid_post_a: Activations from first image [1, hidden_dim]
        resid_post_b: Activations from second image [1, hidden_dim]
        interpolation_layer: Which layer to interpolate at
        n_steps: Number of interpolation steps
        device: Device to run on
        freeze_attention: Whether to freeze attention outputs
        freeze_mlp: Whether to freeze MLP outputs
    
    Returns:
        Dict with keys like 'layer{i}_output' -> tensors of shape [n_steps, hidden_dim]
        and 'logits' -> tensor of shape [n_steps, num_classes]
    """
    if 'similar1' not in shared_images or 'similar2' not in shared_images or 'distinct' not in shared_images:
        raise ValueError("shared_images must contain 'similar1', 'similar2', and 'distinct' keys with image URLs.")
    
    n_layers = get_n_layers_from_model(model)
    # Get activations at interpolation layer for both images
    resid_a = resid_post_a[f'layer{interpolation_layer}_resid_post']  # [1, hidden_dim]
    resid_b = resid_post_b[f'layer{interpolation_layer}_resid_post']  # [1, hidden_dim]
    assert not (1e-5 < (resid_a - resid_b).sum() < 1e-5), "Resid_post activations for both images are identical."
    
    # Compute SLERP interpolations
    resid_a_device = resid_a.to(device)
    resid_b_device = resid_b.to(device)
    alphas = torch.linspace(0, 1, n_steps, device=device)
    
    interpolated_activations = torch.stack([
        slerp_rescale(resid_a_device, resid_b_device, alpha.item()).squeeze(0)
        for alpha in alphas
    ])  # [n_steps, hidden_dim]
    
    # Storage for collected activations (keep device copy for injection)
    activations = {f'layer{interpolation_layer}_resid_post': interpolated_activations.cpu().clone()}
    
    # Hook to inject interpolated activations at the interpolation layer
    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            output = list(output)
            # Replace CLS token with interpolated values
            output[0][:, 0, :] = interpolated_activations
            return tuple(output)
        else:
            output[:, 0, :] = interpolated_activations
            return output
    
    # Hook to collect attention outputs
    def create_attn_hook(target_layer):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]  # attention_output
            else:
                attn_output = output
            
            # Freeze if requested
            if freeze_attention:
                mean_activation = attn_output[:, 0, :].mean(dim=0, keepdim=True)
                attn_output[:, 0, :] = mean_activation.expand(attn_output.shape[0], -1)
            
            # Store CLS token activation [n_steps, hidden_dim]
            activations[f'layer{target_layer}_attn_out'] = attn_output[:, 0, :].cpu().unsqueeze(1).clone()
            return output
        return hook_fn
    
    # Hook to collect resid_mid (after attention, before MLP)
    def create_resid_mid_hook(target_layer):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store CLS token activation [n_steps, hidden_dim]
            activations[f'layer{target_layer}_resid_mid'] = hidden_states[:, 0, :].cpu().unsqueeze(1).clone()
            return output
        return hook_fn
    
    # Hook to collect MLP intermediate (mlp_post) - output of intermediate layer before activation
    def create_mlp_post_hook(target_layer):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                mlp_post = output[0]
            else:
                mlp_post = output
            # Store CLS token activation [n_steps, hidden_dim]
            activations[f'layer{target_layer}_mlp_post'] = mlp_post[:, 0, :].cpu().unsqueeze(1).clone()
            return output
        return hook_fn
    
    # Hook to collect MLP outputs
    def create_mlp_hook(target_layer):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                mlp_output = output[0]
            else:
                mlp_output = output
            
            # Freeze if requested
            if freeze_mlp:
                mean_activation = mlp_output[:, 0, :].mean(dim=0, keepdim=True)
                mlp_output[:, 0, :] = mean_activation.expand(mlp_output.shape[0], -1)
            
            # Store CLS token activation [n_steps, hidden_dim]
            activations[f'layer{target_layer}_mlp_out'] = mlp_output[:, 0, :].cpu().unsqueeze(1).clone()
            return output
        return hook_fn
    
    # Hook to collect resid_post (after MLP, final layer output)
    def create_resid_post_hook(target_layer):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store CLS token activation [n_steps, hidden_dim]
            activations[f'layer{target_layer}_resid_post'] = hidden_states[:, 0, :].cpu().unsqueeze(1).clone()
            return output
        return hook_fn
    
    # Register hooks
    hooks = []
    
    # Injection hook at interpolation layer - inject the interpolated CLS token activation
    injection_hook = model.vit.encoder.layer[interpolation_layer].register_forward_hook(inject_hook)
    hooks.append(injection_hook)
    
    # Collection hooks for subsequent layers
    for i in range(interpolation_layer, n_layers):
        layer = model.vit.encoder.layer[i]
        
        # Hook attention output
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'output'):
            attn_hook = layer.attention.output.register_forward_hook(create_attn_hook(i))
            hooks.append(attn_hook)
        elif hasattr(layer, 'attention'):
            # Some ViT models have attention output directly
            attn_hook = layer.attention.register_forward_hook(create_attn_hook(i))
            hooks.append(attn_hook)
        
        # Hook resid_mid (after attention, before MLP) - this is the layer output after attention
        # We'll capture it from the layer's intermediate output
        if hasattr(layer, 'layernorm_after'):
            resid_mid_hook = layer.layernorm_after.register_forward_hook(create_resid_mid_hook(i))
            hooks.append(resid_mid_hook)
        
        # Hook MLP intermediate (mlp_post) - output of intermediate.dense
        if hasattr(layer, 'intermediate'):
            if hasattr(layer.intermediate, 'dense'):
                mlp_post_hook = layer.intermediate.dense.register_forward_hook(create_mlp_post_hook(i))
                hooks.append(mlp_post_hook)
        
        # Hook MLP output
        if hasattr(layer, 'output'):
            mlp_hook = layer.output.register_forward_hook(create_mlp_hook(i))
            hooks.append(mlp_hook)
        elif hasattr(layer, 'mlp'):
            if hasattr(layer.mlp, 'output'):
                mlp_hook = layer.mlp.output.register_forward_hook(create_mlp_hook(i))
                hooks.append(mlp_hook)
            else:
                mlp_hook = layer.mlp.register_forward_hook(create_mlp_hook(i))
                hooks.append(mlp_hook)
        
        # Hook layer output (resid_post) - final output of the layer
        layer_hook = layer.register_forward_hook(create_resid_post_hook(i))
        hooks.append(layer_hook)
    
    # Create batched inputs by repeating the image inputs

    image_similar1 = Image.open(requests.get(shared_images['similar1'], stream=True).raw)
    image_similar2 = Image.open(requests.get(shared_images['similar2'], stream=True).raw)
    image_distinct = Image.open(requests.get(shared_images['distinct'], stream=True).raw)

    # limitation: linear interpolation. Better synthesis should be used.

    inputs_similar1 = processor(images=image_similar1, return_tensors="pt").to(device)
    inputs_similar2 = processor(images=image_similar2, return_tensors="pt").to(device)
    inputs_distinct = processor(images=image_distinct, return_tensors="pt").to(device)

    # average of similar1, similar2, and distinct
    pixel_avg = (inputs_similar1['pixel_values'] + inputs_similar2['pixel_values'] + inputs_distinct['pixel_values']) / 3.0
    inputs = {'pixel_values': pixel_avg}

    batched_inputs = {k: v.repeat(n_steps, 1, 1, 1) if len(v.shape) == 4 else v.repeat(n_steps, 1) if len(v.shape) == 2 else v.repeat(n_steps)
                     for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**batched_inputs)
        logits = outputs.logits
    
    activations['logits'] = logits.cpu().clone()
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up
    del interpolated_activations, logits, inputs_similar1, inputs_similar2, inputs_distinct, inputs, batched_inputs, pixel_avg
    torch.cuda.empty_cache()
    
    return activations

def interpolate_layer_wrapper(model_type: str, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Wrapper to interpolate activations at a specified layer for different model types.
    
    :param model_type: Type of model ('hooked_transformer' or 'vit')
    :type model_type: str
    :param kwargs: Model-specific arguments (model, processor, resid_post_a, resid_post_b, etc.)
    :return: Dictionary mapping layer names to interpolated activation tensors
    :rtype: Dict[str, torch.Tensor]
    :raises ValueError: If model_type is not supported
    """
    if model_type not in ['hooked_transformer', 'vit']:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if model_type == 'hooked_transformer':
        return interpolate_resid_post_layer(**kwargs)
    
    if model_type == 'vit':
        return interpolate_vit_layer(**kwargs)

def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--freeze_attention', action='store_true', help='Freeze attention outputs to first step')
    parser.add_argument('--freeze_mlp', action='store_true', help='Freeze MLP outputs to first step')
    parser.add_argument('--interpolate_only_first_layer', action='store_true', help='Only interpolate at the first layer for less data')

    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet'], required=True, help='Type of model to use (hooked_transformer or vit)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image'], required=True, help='Type of data type to use (text or image)')

    args = parser.parse_args()

    MODEL_NAME = config['model_names'][args.model_type]

    # if model type is vit, choose the model set default in config

    print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")
    print(f"Freeze attention: {args.freeze_attention} | Freeze MLP: {args.freeze_mlp}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(MODEL_NAME)
    model = model.to(device)
    n_layers = get_n_layers_from_model(model)
    print(f"Loaded {n_layers}-layer model on {device}")

    output_dir = f"./activations/{MODEL_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    # configure parameters in advance for convenience
    model_type = args.model_type
    if args.data_type == 'text':
        SHARED_CONTEXT = config['text']['shared_context']
        PAIRS = config['text']['token_pairs']
        params_collect = {
            'model': model,
            'shared_context': SHARED_CONTEXT,
            'variable_data': None,  # placeholder for token
            'device': device
        }
        params_interpolate = {
            'model': model,
            'shared_context': SHARED_CONTEXT,
            'resid_post_a': None,
            'resid_post_b': None,
            'interpolation_layer': None,
            'n_steps': N_STEPS,
            'device': device,
            'freeze_attention': False,
            'freeze_mlp': False,
        }

        # for path construction
        SHARED_ID = SHARED_CONTEXT
        PAIRS_IDS = PAIRS

    elif args.data_type == 'image':
        SHARED_IMAGES = config["image"]["shared_images"]
        PAIRS = config['image']['image_pairs']
        processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        params_collect = {
            'model': model,
            'processor': processor,
            'variable_data': None,  # placeholder for image_url
            'device': device
        }
        params_interpolate = {
            'model': model,
            'processor': processor,
            'shared_images': SHARED_IMAGES,
            'resid_post_a': None,
            'resid_post_b': None,
            'interpolation_layer': None,
            'n_steps': N_STEPS,
            'device': device,
            'freeze_attention': False,
            'freeze_mlp': False,
        }

        # for path construction
        SHARED_ID = config['image']['shared_image_id']
        PAIRS_IDS = config['image']['pairs_ids']

    for i, pair in enumerate(PAIRS):
        print(f"\nProcessing {pair}")

        # Collect reference activations for both tokens
        reference_activations = {}
        for idx, variable_data in enumerate(pair):
            params_collect['variable_data'] = variable_data
            reference_activations[f'token_{idx}'] = collect_original_activations_wrapper(
                model_type, **params_collect
            )

        # Interpolate at each layer
        freeze_suffix = ""
        if args.freeze_attention:
            freeze_suffix = "_freeze_attn"
        elif args.freeze_mlp:
            freeze_suffix = "_freeze_mlp"

        if args.interpolate_only_first_layer:
            if args.model_type == 'vit':
                layer_to_interpolate = 1  # ViT's CLS token is initialized the same for layer 0. Therefore, interpolating between two IDENTICAL CLS tokens at layer 0 is meaningless.
            else:
                layer_to_interpolate = 0
            pbar = tqdm([layer_to_interpolate], desc=f"Interpolating {pair}{freeze_suffix}")
        else:
            pbar = tqdm(range(n_layers), desc=f"Interpolating {pair}{freeze_suffix}")

        for interpolation_layer in pbar:
            
            params_interpolate['resid_post_a'] = reference_activations['token_0']
            params_interpolate['resid_post_b'] = reference_activations['token_1']
            params_interpolate['interpolation_layer'] = interpolation_layer
            params_interpolate['freeze_attention'] = args.freeze_attention
            params_interpolate['freeze_mlp'] = args.freeze_mlp

            interpolated_activations = interpolate_layer_wrapper(
                model_type=model_type,
                **params_interpolate
            )

            # Save to disk
            filepath = construct_filepath(MODEL_NAME, SHARED_ID, interpolation_layer, PAIRS_IDS[i], N_STEPS, freeze_suffix)
            torch.save(interpolated_activations, filepath)

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()