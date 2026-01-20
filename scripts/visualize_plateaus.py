#!/usr/bin/env python3
"""
"""

from typing import Union
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
from model import ResNetMLP, ResNetMLPSkeleton

config = load_config()

def collect_activation_maps_resid_post(model: Union[ResNetMLP, ResNetMLPSkeleton], data, device, input_layer_idx: int, output_layer_idx: int) -> Dict[str, torch.Tensor]:
    """
    Collect activation maps (input-output pairs) for a given toy model.
    Args:
        model: The toy model to use.
        data: The data to use.
        device: The device to use.
        input_layer_key: The key of the input layer.
        output_layer_key: The key of the output layer.
    Returns:
        A dictionary of activation maps.
    """
    
    activations = {}
    
    def create_hook(layer_idx):
        def hook(module, input, output):
            activations[f'layer{layer_idx}_resid_post'] = output.cpu().clone()
        return hook
    
    layers = list(model.blocks) + [model.hook_input, model.hook_embed]
    
    hooks = []
    hooks.append(layers[input_layer_idx].register_forward_hook(create_hook(input_layer_idx)))
    hooks.append(layers[output_layer_idx].register_forward_hook(create_hook(output_layer_idx)))

    model(data)

    for hook in hooks:
        hook.remove()

    return activations

def generate_uniform_data_around_point()

def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet', 'toy_resnet'], required=True, help='Type of model to use (hooked_transformer or vit)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image', 'class_spiral'], required=True, help='Type of data type to use (text or image)')

    args = parser.parse_args()

    MODEL_NAME = config['model_names'][args.model_type]
    N_POINTS = config['visualization']['n_points']

    # if model type is vit, choose the model set default in config

    print(f"Model: {MODEL_NAME} | N_POINTS: {N_POINTS}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_type == 'toy_resnet':
        model, _ = load_model_from_checkpoint(MODEL_NAME)
        n_layers = len(model.blocks)
    else:
        raise ValueError(f"Model type {args.model_type} currently not supported")

    model = model.to(device)
    print(f"Loaded {n_layers}-layer model on {device}")

    output_dir = f"./act_maps/{args.data_type}/{MODEL_NAME}"
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
            elif args.model_type == 'toy_resnet':
                layer_to_interpolate = -2 # hook_input is layer -2
            else:
                layer_to_interpolate = 0
            pbar = tqdm([layer_to_interpolate], desc=f"Interpolating {pair}{freeze_suffix}")
        else:   
            if args.model_type == 'toy_resnet':
                pbar = tqdm(range(-2, n_layers), desc=f"Interpolating {pair}{freeze_suffix}")
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