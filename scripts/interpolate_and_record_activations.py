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
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import ResNetForImageClassification

from PIL import Image
import sys
sys.path.append('./scripts')
sys.path.append('./train')
from utils import load_model, load_config, load_image, slerp_rescale, linear_rescale, construct_filepath, get_n_layers_from_model, load_model_from_checkpoint

from model import ResNetMLP

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

#TODO: create util func to get layers from ViT
#TODO: create util func to load image from url


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

def collect_original_activations_vit(model: ViTForImageClassification, processor: AutoImageProcessor, variable_data: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Collect resid_post activations of all patch tokens at all layers.

    Returns dict with keys like 'layer{i}_resid_post' -> tensor of shape [1, hidden_dim]
    """
    image = load_image(variable_data)
    inputs = processor(images=image, return_tensors="pt").to(device)

    activations = {}

    def create_collection_hook(layer_idx):
        """Create hook to collect all patch token activations at each layer."""
        def hook_fn(module, input, output):
            # For ViT, output is typically (batch_size, num_patches + 1, hidden_dim)
            # We take the non-CLS tokens (all tokens except the first) activation
            if isinstance(output, tuple):
                output = output[0]
            activations[f'layer{layer_idx}_resid_post'] = output[:, :, :].cpu().clone()
        return hook_fn

    hooks = []

    if hasattr(model, 'vit'):
        layers = model.vit.encoder.layer
    else:
        layers = model.encoder.layer

    for i, layer in enumerate(layers):
        hook = layer.register_forward_hook(create_collection_hook(i))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    del inputs, _
    torch.cuda.empty_cache() if device == 'cuda' else None

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations

def collect_original_activations_resnet(model: ResNetForImageClassification, processor: AutoImageProcessor, variable_data: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Collect resid_post activations from ResNet layers.
    For ResNet, we collect spatial-pooled features since ResNet outputs are feature maps.

    Returns dict with keys like 'layer{i}_resid_post' -> tensor of shape [1, channels, height, width]
    """
    image = load_image(variable_data)
    
    inputs = processor(images=image, return_tensors="pt").to(device)

    activations = {}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    layer_ids = list(range(len(outputs.hidden_states)))    # outputs.hidden_states is a tuple of all layer outputs
    
    for idx in layer_ids:
        activations[f'layer{idx}_resid_post'] = outputs.hidden_states[idx].cpu().clone()
        
    # Clean up
    del inputs, outputs
    torch.cuda.empty_cache()

    return activations

def collect_original_activations_toy_resnet(model: ResNetMLP, variable_data: torch.Tensor, device: str) -> Dict[str, torch.Tensor]:
    """
    Collect activations from a toy model.
    """
    inputs = variable_data
    
    activations = {}

    def create_collection_hook(layer_idx):
        """Create hook to collect all patch token activations at each layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[f"layer{layer_idx}_resid_post"] = output.cpu().clone()
        return hook_fn

    hooks = []

    for layer_idx, residual_block in enumerate(model.blocks):
        hooks.append(residual_block.hook_resid_post.register_forward_hook(create_collection_hook(layer_idx)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    # Clean up
    del inputs, outputs
    torch.cuda.empty_cache()

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
    if model_type == 'hooked_transformer':
        return collect_original_activations_hooked_transformer(**kwargs)

    elif model_type == 'vit':
        return collect_original_activations_vit(**kwargs)
    
    elif model_type == 'resnet':
        return collect_original_activations_resnet(**kwargs)

    elif model_type == 'toy_resnet':
        return collect_original_activations_toy_resnet(**kwargs)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
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
    shared_image: str, # url_to_image
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
    Interpolates non-CLS token activations but collect only CLS token activations.
    
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
    
    n_layers = get_n_layers_from_model(model)
    # Get activations at interpolation layer for both images
    resid_a = resid_post_a[f'layer{interpolation_layer}_resid_post']  # [1, 216, hidden_dim]    
    resid_b = resid_post_b[f'layer{interpolation_layer}_resid_post']  # [1, 216, hidden_dim]
    
    # Compute SLERP interpolations
    resid_a_device = resid_a.to(device)
    resid_b_device = resid_b.to(device)
    alphas = torch.linspace(0, 1, n_steps, device=device)
    
    interpolated_activations = torch.stack([
        slerp_rescale(resid_a_device, resid_b_device, alpha.item()).squeeze(0)
        for alpha in alphas
    ])  # [n_steps, hidden_dim]
    
    # Storage for collected activations
    activations = {f'layer{interpolation_layer}_resid_post': interpolated_activations.cpu().clone()}
    
    # Hook to inject interpolated activations
    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            output = list(output)
            # Replace non-CLS tokens with interpolated values
            output[0][:, :, :] = interpolated_activations
            return tuple(output)
        else:
            output[:, :, :] = interpolated_activations
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
                mean_activation = attn_output[:, :, :].mean(dim=0, keepdim=True)
                attn_output[:, :, :] = mean_activation.expand(attn_output.shape[0], -1)
            
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
                mean_activation = mlp_output[:, :, :].mean(dim=0, keepdim=True)
                mlp_output[:, :, :] = mean_activation.expand(mlp_output.shape[0], -1)
            
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
    
    # get layers
    layers = []
    if hasattr(model, 'vit'):
        layers = model.vit.encoder.layer
    elif hasattr(model, 'encoder'):
        layers = model.encoder.layer
    else:
        raise ValueError("Model does not have recognizable transformer layers for ViT.")
    
    # Register hooks
    hooks = []
    
    # Injection hook at interpolation layer - inject the interpolated CLS token activation
    injection_hook = layers[interpolation_layer].register_forward_hook(inject_hook)
    hooks.append(injection_hook)
    
    # Collection hooks for subsequent layers
    for i in range(interpolation_layer, n_layers):
        layer = layers[i]
        
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
    image = load_image(shared_image)
    inputs = processor(images=image, return_tensors="pt").to(device)

    batched_inputs = {k: v.repeat(n_steps, 1, 1, 1) if len(v.shape) == 4 else v.repeat(n_steps, 1) if len(v.shape) == 2 else v.repeat(n_steps)
                     for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**batched_inputs)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs.last_hidden_state  # Fallback
    
    activations['logits'] = logits.cpu().clone()
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up
    del interpolated_activations, logits, inputs, batched_inputs
    torch.cuda.empty_cache()
    
    return activations

def interpolate_toy_resnet_layer(
    model: ResNetMLP,
    resid_post_a: Dict[str, torch.Tensor],
    resid_post_b: Dict[str, torch.Tensor],
    interpolation_layer: int,
    n_steps: int,
    device: str,
    freeze_attention: bool = False,
    freeze_mlp: bool = False,
) -> Dict[str, torch.Tensor]:
    
    n_layers = len(model.blocks)
    # Get activations at interpolation layer for both images
    resid_a = resid_post_a[f'layer{interpolation_layer}_resid_post']  # [1, 216, hidden_dim]    
    resid_b = resid_post_b[f'layer{interpolation_layer}_resid_post']  # [1, 216, hidden_dim]
    
    # Compute SLERP interpolations
    resid_a_device = resid_a.to(device)
    resid_b_device = resid_b.to(device)
    alphas = torch.linspace(0, 1, n_steps, device=device)
    
    # LINEAR INTERPOLATION
    interpolated_activations = torch.stack([
        linear_rescale(resid_a_device, resid_b_device, alpha.item()).squeeze(0)
        for alpha in alphas
    ])  # [n_steps, hidden_dim]
    
    # Storage for collected activations
    activations = {f'layer{interpolation_layer}_resid_post': interpolated_activations.cpu().clone()}
    
    # Hook to inject interpolated activations
    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            output = list(output)
            # Replace non-CLS tokens with interpolated values
            output[0][:, :] = interpolated_activations
            return tuple(output)
        else:
            output[:, :] = interpolated_activations
            return output
    
    # Hook to collect resid_post (after MLP, final layer output)
    def create_resid_post_hook(target_layer):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store activation [n_steps, hidden_dim]
            activations[f'layer{target_layer}_resid_post'] = hidden_states.cpu().clone()
            return output
        return hook_fn
    
    # get layers
    layers = model.blocks

    # Register hooks
    hooks = []
    
    # Injection hook at interpolation layer - inject the interpolated CLS token activation
    injection_hook = layers[interpolation_layer].register_forward_hook(inject_hook)
    hooks.append(injection_hook)
    
    # Collection hooks for subsequent layers
    for i in range(interpolation_layer, n_layers):
        layer = layers[i]
        
        # Hook layer output (resid_post) - final output of the layer
        layer_hook = layer.register_forward_hook(create_resid_post_hook(i))
        hooks.append(layer_hook)
    
    # Create dummy inputs
    batched_inputs = torch.zeros((n_steps, model.input_layer.in_features), device=device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(batched_inputs)
    
    activations['logits'] = logits.cpu().clone()
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up
    del interpolated_activations, logits, batched_inputs
    torch.cuda.empty_cache()
    
    return activations

def slerp_cnn(v0: torch.Tensor, v1: torch.Tensor, n_steps: int, device: str) -> torch.Tensor:
    """
    SLERP wrapper for ResNet feature maps.
    Treats the entire feature map as a single vector and performs spherical linear interpolation.
    
    Args:
        v0: First feature map [1, C, H, W]
        v1: Second feature map [1, C, H, W]
        n_steps: Number of interpolation steps
        device: Device to perform computation on
    
    Returns:
        Interpolated feature maps [n_steps, C, H, W]
    """
    if len(v0.shape) != 4:
        raise ValueError(f"Expected 4D tensor [1, C, H, W], got shape {v0.shape}")
    
    # v0, v1 shape: [1, C, H, W]
    # Save dimensions
    bs, c, h, w = v0.shape
    
    # Flatten: [1, C*H*W]
    v0_flat = v0.view(bs, -1).to(device)
    v1_flat = v1.view(bs, -1).to(device)
    
    alphas = torch.linspace(0, 1, n_steps, device=device)
    
    # perform SLERP in flattened space
    # output shape: [n_steps, C*H*W]
    interpolated_flat = torch.stack([
        slerp_rescale(v0_flat, v1_flat, alpha.item()).squeeze(0)
        for alpha in alphas
    ])
    
    # Clean up intermediate tensors
    del v0_flat, v1_flat, alphas
    
    # Reshape back to original shape: [n_steps, C, H, W]
    return interpolated_flat.view(n_steps, c, h, w)

def lerp_cnn(v0: torch.Tensor, v1: torch.Tensor, n_steps: int, device: str) -> torch.Tensor:
    """
    Linear interpolation wrapper for ResNet feature maps.
    Treats the entire feature map as a single vector and performs linear interpolation.
    
    Args:
        v0: First feature map [1, C, H, W]
        v1: Second feature map [1, C, H, W]
        n_steps: Number of interpolation steps
        device: Device to perform computation on
    
    Returns:
        Interpolated feature maps [n_steps, C, H, W]
    """
    if len(v0.shape) != 4:
        raise ValueError(f"Expected 4D tensor [1, C, H, W], got shape {v0.shape}")
    
    # v0, v1 shape: [1, C, H, W]
    # Save dimensions
    bs, c, h, w = v0.shape
    
    # Flatten: [1, C*H*W]
    v0_flat = v0.view(bs, -1).to(device)
    v1_flat = v1.view(bs, -1).to(device)
    
    alphas = torch.linspace(0, 1, n_steps, device=device)
    
    # perform linear interpolation in flattened space
    # output shape: [n_steps, C*H*W]
    interpolated_flat = torch.stack([
        linear_rescale(v0_flat, v1_flat, alpha.item()).squeeze(0)
        for alpha in alphas
    ])
    
    # Clean up intermediate tensors
    del v0_flat, v1_flat, alphas
    
    # Reshape back to original shape: [n_steps, C, H, W]
    return interpolated_flat.view(n_steps, c, h, w)

def interpolate_resnet_layer(
    model: ResNetForImageClassification,
    processor: AutoImageProcessor,
    shared_image: str,
    resid_post_a: Dict[str, torch.Tensor],
    resid_post_b: Dict[str, torch.Tensor],
    interpolation_layer: int,
    n_steps: int,
    device: str,
    freeze_attention: bool = False, # unnecessary for ResNet
    freeze_mlp: bool = False        # unnecessary for ResNet
) -> Dict[str, torch.Tensor]:
    """
    Interpolate resid_post at specified stage and record all downstream activations.

    Returns dict with keys like 'layer{i}_{hook_name}' -> tensors of shape [n_steps, C, H, W]
    and 'logits' -> tensor of shape [n_steps, num_classes]
    """
    
    # Access ResNet model components with better error handling
    if hasattr(model, 'resnet'):
        resnet = model.resnet
    else:
        resnet = model
    
    # Build list of modules - handle different ResNet structures
    modules_list = [resnet.embedder] + list(resnet.encoder.stages)
    
    n_layers = len(modules_list)
    
    if interpolation_layer >= n_layers:
        raise ValueError(f"Interpolation layer {interpolation_layer} is out of bounds for ResNet (max {n_layers-1}).")

    key = f'layer{interpolation_layer}_resid_post'
    
    resid_a = resid_post_a[key]
    resid_b = resid_post_b[key]
    
    # Spherical interpolation for ResNet (Flatten -> Linear Interpolation -> Reshape)
    interpolated_activations = slerp_cnn(resid_a, resid_b, n_steps, device)  # (n_steps, C, H, W)
    
    # Ensure interpolated activations are on device for injection
    interpolated_activations_device = interpolated_activations.to(device)
    
    # Dictionary to store results (store injected layer values on CPU)
    activations = {key: interpolated_activations.cpu().clone()}
    
    # Injection Hook
    def inject_hook(module, input, output):
        # output shape: (batch, C, H, W)
        # interpolated_activations_device shape: [n_steps, C, H, W]
        # Overwrite for batch size (input is repeated n_steps times)
        return interpolated_activations_device

    # Collection Hook
    def create_collection_hook(target_layer_idx):
        def hook_fn(module, input, output):
            activations[f'layer{target_layer_idx}_resid_post'] = output.cpu().clone()
            return output
        return hook_fn

    hooks = []
    
    target_module = modules_list[interpolation_layer]
    hooks.append(target_module.register_forward_hook(inject_hook))
    
    # Register collection hooks for downstream layers
    for i in range(interpolation_layer + 1, n_layers):
        hooks.append(modules_list[i].register_forward_hook(create_collection_hook(i)))

    # Dummy input image
    if shared_image:
        image = load_image(shared_image)
    else:
        # fallback if shared_image is not provided (e.g., random noise image)
        image = Image.new('RGB', (224, 224))
        
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # expand batch size to n_steps
    batched_inputs = {
        k: v.repeat(n_steps, 1, 1, 1) if v.ndim == 4 else v.repeat(n_steps) if v.ndim > 0 else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**batched_inputs)
        
    activations['logits'] = outputs.logits.cpu().clone()

    for hook in hooks:
        hook.remove()
    
    # Clean up
    del interpolated_activations, interpolated_activations_device, batched_inputs, inputs
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
    
    if model_type == 'hooked_transformer':
        return interpolate_resid_post_layer(**kwargs)
    
    elif model_type == 'vit':
        return interpolate_vit_layer(**kwargs)
    
    elif model_type == 'resnet':
        return interpolate_resnet_layer(**kwargs)

    elif model_type == 'toy_resnet':
        return interpolate_toy_resnet_layer(**kwargs)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--freeze_attention', action='store_true', help='Freeze attention outputs to first step')
    parser.add_argument('--freeze_mlp', action='store_true', help='Freeze MLP outputs to first step')
    parser.add_argument('--interpolate_only_first_layer', action='store_true', help='Only interpolate at the first layer for less data')

    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet', 'toy_resnet'], required=True, help='Type of model to use (hooked_transformer or vit)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image', 'class_spiral'], required=True, help='Type of data type to use (text or image)')

    args = parser.parse_args()

    MODEL_NAME = config['model_names'][args.model_type]

    # if model type is vit, choose the model set default in config

    print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")
    print(f"Freeze attention: {args.freeze_attention} | Freeze MLP: {args.freeze_mlp}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_type == 'toy_resnet':
        model, _ = load_model_from_checkpoint(MODEL_NAME)
        n_layers = len(model.blocks)
    else:
        model = load_model(MODEL_NAME)
        n_layers = get_n_layers_from_model(model)

    model = model.to(device)
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
            'freeze_attention': args.freeze_attention,
            'freeze_mlp': args.freeze_mlp,
        }

        # for path construction
        SHARED_ID = SHARED_CONTEXT
        PAIRS_IDS = PAIRS

    elif args.data_type == 'image':
        SHARED_IMAGE = config["image"]["shared_image"]
        PAIRS = config['image']['image_pairs']
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        params_collect = {
            'model': model,
            'processor': processor,
            'variable_data': None,  # placeholder for image_url
            'device': device
        }
        params_interpolate = {
            'model': model,
            'processor': processor,
            'shared_image': SHARED_IMAGE,
            'resid_post_a': None,
            'resid_post_b': None,
            'interpolation_layer': None,
            'n_steps': N_STEPS,
            'device': device,
            'freeze_attention': args.freeze_attention,
            'freeze_mlp': args.freeze_mlp,
        }

        # for path construction
        SHARED_ID = config['image']['shared_image_id']
        PAIRS_IDS = config['image']['pairs_ids']

    elif args.data_type == 'class_spiral':
        PAIRS = config['class_spiral']['pairs']

        params_collect = {
            'model': model,
            'variable_data': None,  # placeholder for pair
            'device': device
        }
        params_interpolate = {
            'model': model,
            'resid_post_a': None,
            'resid_post_b': None,
            'interpolation_layer': None,
            'n_steps': N_STEPS,
            'device': device,
        }

        # for path construction
        SHARED_ID = ''
        PAIRS_IDS = [[f'{num}' for num in pair] for pair in PAIRS]
        PAIRS = [torch.tensor(pair) for pair in PAIRS]

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