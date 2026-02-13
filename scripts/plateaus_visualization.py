#!/usr/bin/env python3
"""
"""

from ctypes.wintypes import POINT
from typing import Union
from test import activations
import torch
import numpy as np
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
from utils import load_config, load_model,load_model_from_checkpoint, construct_filepath, get_n_layers_from_model, LAYER_ARGUMENT_IDX_MAPPING
from model import ResNetMLP, ResNetMLPSkeleton

config = load_config()

METRIC_OPTIONS = ['l2_norm', 'jacobian_determinant_full', 'jacobian_determinant_layerwise_prod']

def collect_activations_resid_post(model: Union[ResNetMLP, ResNetMLPSkeleton], data, source_layer_idx: int, target_layer_idx: int, device) -> Dict[str, torch.Tensor]:
    
    activations = {}
    
    def create_hook(layer_idx):
        def hook(module, input, output):
            activations[f'layer{layer_idx}_resid_post'] = output.cpu().clone()
        return hook
    
    layers = list(model.blocks) + [model.hook_input, model.hook_embed]
    hooks = []
    for layer_idx in range(source_layer_idx, target_layer_idx + 1):
        hooks.append(layers[layer_idx].register_forward_hook(create_hook(layer_idx)))

    with torch.no_grad():
        logits = model(data)

    activations['logits'] = logits.cpu().clone()

    for hook in hooks:
        hook.remove()

    return activations

def generate_uniform_data_around_point(point: torch.Tensor, n_points: int, radius: float) -> torch.Tensor:
    """
    Generate n_points uniformly distributed around a point in the same dimension.
    Args:
        point: [dim]
        n_points: number of points to generate
        radius: maximum distance from the original point
    Returns:
        Tensor of shape [n_points, dim]
    """
    dim = point.shape[0]
    random_directions = torch.randn(n_points, dim)
    random_directions /= torch.norm(random_directions, dim=1, keepdim=True)  # Normalize to unit vectors
    random_distances = torch.rand(n_points) * radius  # Random distances from the original point
    perturbations = random_directions * random_distances.unsqueeze(1)  # Scale by distances
    new_points = point.unsqueeze(0) + perturbations  # Shift by the original point
    return new_points

def compute_jacobian_source_target_toy(model: Union[ResNetMLP, ResNetMLPSkeleton], source_layer_idx: int, target_layer_idx: int, data: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute jacobian from source layer to target layer, without respect to the intermediate layers.
    
    Args:
        model: Toy ResNet
        source_layer_idx: the first layer of the function of which you want to calculate the jacobian. Therefore, the given data should be sampled in the output space/codomain of layer source_layer_idx - 1.
        target_layer_idx: the last layer of the function of which you want to calculate the jacobian. Therefore, a metric is calculated for points in the output space/codomain of this layer.
        data: A batch of points in the domain of the function. Shape: (batch_size, n_dim_source)
        device: target device of the model
    """

    jacobians = []
    batch_size = data.shape[0]
    layers = list(model.blocks) + [model.hook_input, model.hook_embed]  # hook_input is layer -2, hook_embed is layer -1

    def forward_through_intermediate_layers(resid):
        activation = resid.unsqueeze(0)
        for layer_idx in range(source_layer_idx, target_layer_idx):
            activation = layers[layer_idx](activation)
        return activation.squeeze(0)

    for i in range(batch_size):
        resid = data[i].to(device)
        jac = torch.func.jacrev(forward_through_intermediate_layers)(resid) # Tensor: (d_target, d_source)
        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)

def _compute_jacobian_layerwise_toy(model: Union[ResNetMLP, ResNetMLPSkeleton], layer_idx: int, data: torch.Tensor, device: str, activations_of_layer: torch.Tensor) -> torch.Tensor:
    """
    Compute jacobian for single block: ∂(layer_i+1 resid_post) / ∂(layer_i resid_post).
    Toy ResNet: no sequence dimension.

    Args:
        model: Toy ResNet
        layer_idx: the index of the layer for which you want to calculate the jacobian. Therefore, the given data should be sampled in the resid_post space of layer layer_idx.
        data: A batch of points in the domain of the function. Shape: (batch_size, n_dim_source)
        device: target device of the model
        activations_of_layer: Tensor containing the recorded activations at this layer for the entire batch, used for validating the correctness of the jacobian calculation.
    Returns:
        Tensor of shape (batch_size, d_target, d_source) containing the Jacobian matrices
    """
    jacobians = []
    batch_size = data.shape[0]
    layers = list(model.blocks) + [model.hook_input, model.hook_embed]  # hook_input is layer -2, hook_embed is layer -1

    def forward_single_layer(resid):
        activation = resid.unsqueeze(0)
        activation = layers[layer_idx + 1](activation)
        return activation.squeeze(0)

    for instance_idx in range(batch_size):
        resid = data[instance_idx].to(device)
        jac = torch.func.jacrev(forward_single_layer)(resid)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_single_layer(resid)
            recorded_out = activations_of_layer[instance_idx].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"Layerwise validation failed: layer {layer_idx}→{layer_idx+1}, instance {instance_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)

def compute_jacobian_source_target_layerwise_toy(model: Union[ResNetMLP, ResNetMLPSkeleton], source_layer_idx: int, target_layer_idx: int, data: torch.Tensor, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian from source layer to target layer, by multiplying the jacobian of each intermediate layer.
    Args:
        model: Toy ResNet
        source_layer_idx: the first layer of the function of which you want to calculate the jacobian. Therefore, the given data should be sampled in the output space/codomain of layer source_layer_idx - 1.
        target_layer_idx: the last layer of the function of which you want to calculate the jacobian. Therefore, a metric is calculated for points in the output space/codomain of this layer.
        data: A batch of points in the domain of the function. Shape: (batch_size, n_dim_source)
        device: target device of the model
        activations: dict containing the recorded activations at each layer and step, used for validating the correctness of the jacobian calculation. Should contain the keys f'layer{layer_idx}_resid_post' for layer_idx in [source_layer_idx, target_layer_idx] for validating the output of each layer.
    Returns:
        Tensor of shape (n_layers, batch_size, d_target_layer, d_source_layer) containing the Jacobian matrices for each layer. The Jacobian from source_layer_idx to target_layer_idx can be calculated by multiplying the Jacobian matrices along the first dimension.
    """
    
    jacobians = []
    for layer_idx in range(source_layer_idx, target_layer_idx):
        jac = _compute_jacobian_layerwise_toy(model, layer_idx, data, device, activations[f'layer{layer_idx}_resid_post']) # (batch_size, d_target_layer, d_source_layer)
        jacobians.append(jac)
        del jac
        torch.cuda.empty_cache()

    return torch.stack(jacobians) # (n_layers, batch_size, d_target_layer, d_source_layer)    
    

def compute_jacobian_determinant(jacobians: torch.Tensor):
    """
    Compute the determinant of a batch of Jacobian matrices.
    Args:
        jacobians: Tensor of shape (..., d_target, d_source)
    Returns:
        Tensor of shape (...) containing the determinant of each Jacobian matrix
    """
    
    # Regard the last two dimensions as (d_target, d_source)
    # Flatten the layer and batch dimensions to calculate determinant all at once
    dims = jacobians.shape[-2:]
    dims_batch = jacobians.shape[:-2]
    assert dims[0] == dims[1], f"Not a square matrix: Shape of each matrix is {dims}"
    jacobians = jacobians.view(-1, *dims)
    dets = torch.linalg.det(jacobians)    # (dims_batch.prod(), )
    dets = dets.view(*dims_batch)
    return dets

def l2_norm_metric(model, point, source_layer_idx, target_layer_idx, device) -> torch.Tensor:
    """
    Compute the L2 norm between a point activation and a batch of data activations.
    Args:
        point_act: Tensor of shape (1, d_target)
        data_acts: Tensor of shape (n_points, d_target)
    Returns:
        Tensor of shape (n_points,) containing the L2 norm between the point activation and each data activation
    """
    point_act = collect_activations_resid_post(
            model,
            point.to(device).unsqueeze(0), 
            source_layer_idx=source_layer_idx, 
            target_layer_idx=target_layer_idx, 
            device=device
        )[f'layer{target_layer_idx}_resid_post']  # (1, d_target)
    
    data_acts = activations[f'layer{target_layer_idx}_resid_post']  # (n_points, d_target)
    return torch.norm(data_acts - point_act, dim=1)

def jacobian_determinant_metric() -> torch.Tensor:
    
    compute_jacobian_source_target_toy(model, source_layer_idx, target_layer_idx, data, device)

    

def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet', 'toy_resnet'], required=True, help='Type of model to use (hooked_transformer or vit)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image', 'class_spiral'], required=True, help='Type of data type to use (text or image)')

    args = parser.parse_args()
    
    model_type = args.model_type

    MODEL_NAME = config['model_names'][model_type]
    N_POINTS = config['plateau_visualization']['n_points']
    RADIUS = config['plateau_visualization']['radius']
    POINT = torch.tensor(config['plateau_visualization']['point'])  # TODO: Accept non-list like data
    METRIC = config['plateau_visualization']['metric']
    SOURCE_LAYER_IDX = config['plateau_visualization']['source_layer_idx']
    TARGET_LAYER_IDX = config['plateau_visualization']['target_layer_idx']

    try:
        source_layer_idx = int(SOURCE_LAYER_IDX)
    except ValueError:
        source_layer_idx = LAYER_ARGUMENT_IDX_MAPPING[model_type][SOURCE_LAYER_IDX]
    try:
        target_layer_idx = int(TARGET_LAYER_IDX)
    except ValueError:
        target_layer_idx = LAYER_ARGUMENT_IDX_MAPPING[model_type][TARGET_LAYER_IDX]

    # if model type is vit, choose the model set default in config

    print(f"Model: {MODEL_NAME} | N_POINTS: {N_POINTS} | POINT: {POINT} | RADIUS: {RADIUS} | METRIC: {METRIC}")
    if METRIC not in METRIC_OPTIONS:
        raise ValueError(f"Metric {METRIC} not supported. Choose from {METRIC_OPTIONS}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type == 'toy_resnet':
        model, _ = load_model_from_checkpoint(MODEL_NAME)
        n_layers = len(model.blocks)
    else:
        raise ValueError(f"Model type {model_type} currently not supported")

    model = model.to(device)
    print(f"Loaded {n_layers}-layer model on {device}")

    # output_dir = f"./act_maps/{args.data_type}/{MODEL_NAME}"
    # os.makedirs(output_dir, exist_ok=True)


    data = generate_uniform_data_around_point(POINT, N_POINTS, RADIUS)  # (n_points, d_source)
    activations = collect_activations_resid_post(model, data.to(device), source_layer_idx=SOURCE_LAYER_IDX, target_layer_idx=TARGET_LAYER_IDX, device=device)

    if METRIC == 'l2_norm':
        pass
        
        
    

    print("\n=== Complete ===")

if __name__ == "__main__":
    main()