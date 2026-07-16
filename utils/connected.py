# utils/connected.py
import numpy as np
import torch
from scipy import ndimage


def _nd_structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(rank=2, connectivity=1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(rank=2, connectivity=2)
    raise ValueError("connectivity must be 4 or 8")


def split_connected_components(
    mask: torch.Tensor,
    connectivity: int = 8,
    min_size: int = 25,
    closing_iterations: int = 1,
    merge_distance: int = 5,
) -> torch.Tensor:
    """
    Convert the mask to a NumPy bool, run morphological closing and connected-
    component labeling on CPU with SciPy, then stack regions that meet the area
    threshold into an N x H x W array of 0/1 masks.

    Args:
        mask: torch.Tensor of shape [H, W], values 0/1 or other foreground values.
        connectivity: 4 or 8, controls the structuring element for connected components.
        min_size: minimum area of a connected component.
        closing_iterations: number of closing iterations, fills thin cracks.
        merge_distance: dilation distance; components within this distance are merged.

    Returns:
        torch.Tensor of shape [N, H, W]. If no component satisfies the conditions,
        returns shape [0, H, W].
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected mask with shape [H, W], got {mask.shape}")

    H, W = mask.shape
    mask_bool = mask.bool()
    if not mask_bool.any():
        return mask.new_zeros((0, H, W))

    mask_np = mask_bool.detach().cpu().numpy()
    structure = _nd_structure(connectivity)

    if closing_iterations > 0:
        closed_np = ndimage.binary_closing(
            mask_np, structure=structure, iterations=closing_iterations
        )
    else:
        closed_np = mask_np

    if merge_distance > 0:
        kernel = np.ones(
            (2 * merge_distance + 1, 2 * merge_distance + 1), dtype=bool
        )
        merge_np = ndimage.binary_dilation(closed_np, structure=kernel)
    else:
        merge_np = closed_np

    labeled, num = ndimage.label(merge_np, structure=structure)
    components_np = []
    for idx in range(1, num + 1):
        comp = np.logical_and(labeled == idx, closed_np)
        if comp.sum() < min_size:
            continue
        components_np.append(comp)

    if len(components_np) == 0:
        return mask.unsqueeze(0)

    stacked = np.stack(components_np, axis=0)
    return torch.from_numpy(stacked).to(device=mask.device, dtype=mask.dtype)
