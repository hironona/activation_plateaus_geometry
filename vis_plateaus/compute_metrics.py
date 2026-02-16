import torch
from typing import Union, Dict
from tqdm import tqdm
import torch.nn.functional as F

import sys
sys.path.append('./train')
from model import ResNetMLP, ResNetMLPSkeleton


def compute_jacobian_source_target_toy(model: Union[ResNetMLP, ResNetMLPSkeleton], data: torch.Tensor, source_layer_idx: int, target_layer_idx: int, device: str) -> torch.Tensor:
    """
    Compute jacobian from source layer to target layer, without respect to the intermediate layers.

    Args:
        model: Toy ResNet
        data: A batch of points in the domain of the function. Shape: (batch_size, n_dim_source)
        source_layer_idx: the first layer of the function of which you want to calculate the jacobian. Therefore, the given data should be sampled in the output space/codomain of layer source_layer_idx - 1.
        target_layer_idx: the last layer of the function of which you want to calculate the jacobian. Therefore, a metric is calculated for points in the output space/codomain of this layer.
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

    jacobian = torch.func.jacrev(forward_through_intermediate_layers)
    for i in tqdm(range(batch_size)):
        resid = data[i].to(device)
        jacobians.append(jacobian(resid).detach().cpu()) # Tensor: (d_target, d_source)
        del resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)

def _compute_jacobian_layerwise_toy(model: Union[ResNetMLP, ResNetMLPSkeleton], data: torch.Tensor, layer_idx: int, device: str, expected_output: torch.Tensor) -> torch.Tensor:
    """
    Compute jacobian for single block: ∂(layer_i+1 resid_post) / ∂(layer_i resid_post).
    Toy ResNet: no sequence dimension.

    Args:
        model: Toy ResNet
        data: A batch of points at layer layer_idx's resid_post. Shape: (batch_size, hidden_dim)
        layer_idx: the index of the input layer. forward applies layers[layer_idx + 1].
        device: target device of the model
        expected_output: Recorded activations at layer layer_idx+1, for validation. Shape: (batch_size, hidden_dim)
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

    jacobian = torch.func.jacrev(forward_single_layer)
    for instance_idx in tqdm(range(batch_size)):
        resid = data[instance_idx].to(device)
        jac = jacobian(resid)

        # Validate against recorded activations at the OUTPUT layer
        with torch.no_grad():
            computed_out = forward_single_layer(resid)
            recorded_out = expected_output[instance_idx].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"Layerwise validation failed: layer {layer_idx}→{layer_idx+1}, instance {instance_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)

def compute_jacobian_source_target_layerwise_toy(model: Union[ResNetMLP, ResNetMLPSkeleton], source_layer_idx: int, target_layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian from source layer to target layer, by multiplying the jacobian of each intermediate layer.
    Args:
        model: Toy ResNet
        source_layer_idx: the first layer of the function of which you want to calculate the jacobian. Therefore, the given data should be sampled in the output space/codomain of layer source_layer_idx - 1.
        target_layer_idx: the last layer of the function of which you want to calculate the jacobian. Therefore, a metric is calculated for points in the output space/codomain of this layer.
        device: target device of the model
        activations: dict containing the recorded activations at each layer. Must contain
            f'layer{i}_resid_post' for i in [source_layer_idx .. target_layer_idx].
    Returns:
        Tensor of shape (n_mid_layers, n_points, d_target_layer, d_source_layer) containing the Jacobian matrices for each layer. The Jacobian from source_layer_idx to target_layer_idx can be calculated by multiplying the Jacobian matrices along the first dimension.
    """

    jacobians = []
    for layer_idx in range(source_layer_idx, target_layer_idx):
        # Input: activations at layer_idx (the input to this transition)
        step_input = activations[f'layer{layer_idx}_resid_post']
        # Expected output: activations at layer_idx+1 (for validation)
        step_expected_output = activations[f'layer{layer_idx + 1}_resid_post']
        jac = _compute_jacobian_layerwise_toy(model, step_input, layer_idx, device, step_expected_output)
        jacobians.append(jac)
        del jac
        torch.cuda.empty_cache()

    return torch.stack(jacobians) # (n_mid_layers, n_points, d_target_layer, d_source_layer)

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

def compute_frobenius_norm(jacobians: torch.Tensor):
    """
    Compute the Frobenius norm of a batch of Jacobian matrices.
    Args:
        jacobians: Tensor of shape (..., d_target, d_source)
    Returns:
        Tensor of shape (...) containing the Frobenius norm of each Jacobian matrix
    """
    dims = jacobians.shape[-2:]
    dims_batch = jacobians.shape[:-2]
    jacobians = jacobians.view(*dims_batch, -1)
    frob_norms = torch.norm(jacobians, dim=-1)  # (dims_batch...)
    return frob_norms


def compute_volume_change_to_logits(model: Union[ResNetMLP, ResNetMLPSkeleton], data: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute the product of singular values of the Jacobian from last block's resid_post to logits.
    Handles the non-square mapping: final_norm -> relu -> output_layer.

    Args:
        model: Toy ResNet model
        data: Activations at the last block's resid_post. Shape: (n_points, hidden_dim)
        device: target device

    Returns:
        Tensor of shape (n_points,) containing the product of singular values per data point.
    """
    norms = []

    def forward_to_logits(resid):
        x = model.final_norm(resid.unsqueeze(0))
        x = F.relu(x)
        x = model.output_layer(x)
        return x.squeeze(0)

    jac_fn = torch.func.jacrev(forward_to_logits)

    for i in tqdm(range(data.shape[0]), desc="Computing last→logits Jacobian norms"):
        j = jac_fn(data[i].to(device))
        # svdvals = torch.linalg.svdvals(j)
        frob_norms = torch.norm(j.flatten(), dim=0)  # Frobenius norm per output dimension
        # norms.append(svdvals.prod().detach().cpu())
        norms.append(frob_norms.detach().cpu())  # Product of Frobenius norms across output dimensions
        del j
        torch.cuda.empty_cache()

    return torch.stack(norms)  # (n_points,)


def compute_volume_change_input_to_embed(model: Union[ResNetMLP, ResNetMLPSkeleton]) -> float:
    """
    Compute the product of singular values of the input_layer weight matrix.
    For ResNetMLPSkeleton (no input_layer / identity mapping), returns 1.0.

    Returns:
        Scalar: product of singular values of input_layer.weight.
    """
    if hasattr(model, 'input_layer'):
        with torch.no_grad():
            # svdvals = torch.linalg.svdvals(model.input_layer.weight.float())
            frob_norms = torch.norm(model.input_layer.weight.flatten())  # Frobenius norm per output dimension
        # return frob_norms.prod().item()
        return frob_norms.item()
    return 1.0


def l2_norm_metric(model, reference_point, activations: Dict[str, torch.Tensor], source_layer_idx, target_layer_idx, device) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Compute the L2 norm between a reference point's activation and a batch of data activations.

    Args:
        model: Toy ResNet model
        reference_point: Tensor of shape (d_input,) — the reference point in raw input space.
        activations: Dictionary of layer activations for the data batch.
        source_layer_idx: source layer index (int).
        target_layer_idx: target layer index (int or 'logits').
        device: target device.

    Returns:
        Dict with 'reference_point_act', 'predicted_class', and 'metric_values'.
    """
    target_is_logits = (target_layer_idx == 'logits')
    n_blocks = len(model.blocks)
    effective_target = n_blocks - 1 if target_is_logits else target_layer_idx
    target_key = 'logits' if target_is_logits else f'layer{target_layer_idx}_resid_post'

    point_activations = collect_activations_resid_post(
        model,
        reference_point.to(device).unsqueeze(0),
        source_layer_idx=source_layer_idx,
        target_layer_idx=effective_target,
        device=device
    )

    point_act = point_activations[target_key]  # (1, d_target)
    data_act = activations[target_key]

    predicted_class = point_activations['logits'].argmax(dim=1).item()

    results = {
        "reference_point_act": point_act,
        "predicted_class": predicted_class,
        "metric_values": torch.norm(data_act - point_act, dim=1)    # (n_points,)
    }

    return results

def jacobian_determinant_metric(model, activations, source_layer_idx, target_layer_idx, device) -> dict:
    """
    Compute the full Jacobian determinant from source to target.
    Handles non-square mappings (input→embed, last_block→logits) via singular value products.

    Args:
        model: Toy ResNet model
        activations: Dictionary of layer activations for the data batch.
        source_layer_idx: source layer index (int).
        target_layer_idx: target layer index (int or 'logits').
        device: target device.

    Returns:
        Dict with 'metric_values' tensor of shape (n_points,).
    """
    target_is_logits = (target_layer_idx == 'logits')
    n_blocks = len(model.blocks)

    # Determine effective bounds for the square Jacobian region
    has_nonsquare_input = (source_layer_idx <= -2 and hasattr(model, 'input_layer'))
    effective_source = -1 if has_nonsquare_input else source_layer_idx

    # Full Jacobian convention: range(source, target) applies layers[source:target]
    # To go through all blocks from embed: target = n_blocks
    square_target = n_blocks if target_is_logits else target_layer_idx

    data = activations[f'layer{effective_source}_resid_post']
    jacobians = compute_jacobian_source_target_toy(model, data, effective_source, square_target, device)  # (n_points, d, d)

    # assert jacobians.shape[-1] == jacobians.shape[-2], f"Jacobian shape is not square: {jacobians.shape}"
    # metric = compute_jacobian_determinant(jacobians)  # (n_points,)
    metric = compute_frobenius_norm(jacobians)  # (n_points,)

    # Multiply by non-square volume changes
    # TODO: DO I really need this
    # if has_nonsquare_input:
    #     input_volume = compute_volume_change_input_to_embed(model)
    #     metric = metric * input_volume

    if target_is_logits:
        last_block_data = activations[f'layer{n_blocks - 1}_resid_post']
        logits_volume = compute_volume_change_to_logits(model, last_block_data, device)
        # logits_volume = logits_volume / torch.max(logits_volume)  # Frobenius norm across output dimensions
        metric = metric * logits_volume

    return {"metric_values": metric}

def jacobian_determinant_layerwise_prod_metric(model, activations, source_layer_idx, target_layer_idx, device) -> dict:
    """
    Compute the product of layerwise Jacobian determinants from source to target.
    Handles non-square mappings (input→embed, last_block→logits) via singular value products.

    Args:
        model: Toy ResNet model
        activations: Dictionary of layer activations for the data batch.
        source_layer_idx: source layer index (int).
        target_layer_idx: target layer index (int or 'logits').
        device: target device.

    Returns:
        Dict with 'metric_values' tensor of shape (n_points,).
    """
    target_is_logits = (target_layer_idx == 'logits')
    n_blocks = len(model.blocks)

    has_nonsquare_input = (source_layer_idx <= -2 and hasattr(model, 'input_layer'))
    effective_source = -1 if has_nonsquare_input else source_layer_idx

    # Layerwise convention: range(source, target) with each step applying layers[idx+1]
    square_target = n_blocks - 1 if target_is_logits else target_layer_idx

    jacobians = compute_jacobian_source_target_layerwise_toy(model, effective_source, square_target, device, activations)  # (n_mid_layers, n_points, d, d)

    # assert jacobians.shape[-1] == jacobians.shape[-2], f"Jacobian shape is not square: {jacobians.shape}"

    # det = compute_jacobian_determinant(jacobians)  # (n_mid_layers, n_points)
    frob_norms = compute_frobenius_norm(jacobians)  # (n_mid_layers, n_points)
    metric = torch.prod(frob_norms, dim=0)  # (n_points,)

    # Multiply by non-square volume changes
    # TODO: DO I really need this
    # if has_nonsquare_input:
    #     input_volume = compute_volume_change_input_to_embed(model)
    #     metric = metric * input_volume

    if target_is_logits:
        last_block_data = activations[f'layer{n_blocks - 1}_resid_post']
        logits_volume = compute_volume_change_to_logits(model, last_block_data, device)
        # logits_volume = logits_volume / torch.max(logits_volume)  # Frobenius norm across output dimensions
        metric = metric * logits_volume

    return {"metric_values": metric}