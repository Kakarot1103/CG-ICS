# encoding: utf-8

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

ColorType = Union[Sequence[int], Sequence[float]]
BoxType = Union[Sequence[float], np.ndarray]
PolygonType = Union[Sequence[float], Sequence[Sequence[float]], np.ndarray]
MaskType = Union[np.ndarray, "torch.Tensor", Image.Image]


def _to_numpy_image(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convert image to uint8 numpy array.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image.copy()


def _normalize_color(color: ColorType) -> Tuple[int, int, int]:
    """
    Convert color with value range [0, 1] or [0, 255] to RGB uint8 tuple.
    """
    arr = np.asarray(color, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Invalid color format, expected length 3 but got {arr.size}")
    if arr.max() <= 1:
        arr = arr * 255
    arr = arr.astype(np.uint8)
    return int(arr[0]), int(arr[1]), int(arr[2])


def _prepare_palette(color_template: Optional[Iterable[ColorType]], num: int) -> List[Tuple[int, int, int]]:
    palette = []
    if color_template is not None:
        for color in color_template:
            palette.append(_normalize_color(color))
    for _ in range(num - len(palette)):
        palette.append(tuple(int(x) for x in np.random.randint(0, 256, size=3)))
    return palette


def _normalize_boxes(boxes: Iterable[BoxType]) -> List[Tuple[int, int, int, int]]:
    normalized = []
    for box in boxes:
        arr = np.asarray(box, dtype=float).reshape(-1)
        if arr.size != 4:
            raise ValueError(f"bbox should have 4 values (x0,y0,x1,y1), but got {arr.size}")
        x0, y0, x1, y1 = [int(round(v)) for v in arr.tolist()]
        normalized.append((x0, y0, x1, y1))
    return normalized


def _normalize_polygons(polygons: Iterable[PolygonType]) -> List[List[np.ndarray]]:
    normalized = []
    for instance_polygons in polygons:
        instance_list = []
        for poly in instance_polygons:
            arr = np.asarray(poly, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            if arr.shape[1] != 2:
                raise ValueError(f"polygon coordinates should be Nx2, but got {arr.shape}")
            instance_list.append(arr.astype(np.int32))
        normalized.append(instance_list)
    return normalized


def _normalize_masks(
    masks: Iterable[MaskType], target_shape: Tuple[int, int]
) -> List[np.ndarray]:
    h, w = target_shape
    normalized = []
    for mask in masks:
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if hasattr(mask, "detach"):
            mask = mask.detach()
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask_arr = np.asarray(mask)
        if mask_arr.shape[:2] != (h, w):
            raise ValueError(
                f"mask size {mask_arr.shape[:2]} does not match image size {(h, w)}, please align first"
            )
        normalized.append((mask_arr > 0).astype(np.uint8))
    return normalized


def _mask_to_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, 0, 0
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    return int(x0), int(y0), int(x1), int(y1)


def _mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(np.int32) for c in contours if c.size >= 6]


def _number_to_string(n: int) -> str:
    chars = []
    while n:
        n, remainder = divmod(n - 1, 26)
        chars.append(chr(97 + remainder))
    return "".join(reversed(chars)) or "a"


def _draw_box_number(
    canvas: np.ndarray, box: Tuple[int, int, int, int], text: str, color: Tuple[int, int, int]
):
    x0, y0, x1, y1 = box
    pos = (x0, y0 - 4 if y0 - 4 > 0 else y0 + 15)
    cv2.putText(
        canvas,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def _draw_mask_number(canvas: np.ndarray, mask: np.ndarray, text: str, color: Tuple[int, int, int]):
    padded = np.pad(mask, ((1, 1), (1, 1)), "constant")
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 0)[1:-1, 1:-1]
    max_dist = float(dist.max())
    if max_dist <= 0:
        return
    ys, xs = np.where(dist == max_dist)
    pos = (int(xs[len(xs) // 2]), int(ys[len(ys) // 2]))
    cv2.putText(
        canvas,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def _validate_lengths(lengths: List[int]):
    non_zero = [l for l in lengths if l is not None]
    if not non_zero:
        return 0
    if len(set(non_zero)) > 1:
        raise ValueError(f"Inconsistent bbox/mask counts: {non_zero}")
    return non_zero[0]


def draw_instances_viz(
    image: Union[np.ndarray, Image.Image],
    color_template: Optional[Iterable[ColorType]] = None,
    boxes: Optional[Iterable[BoxType]] = None,
    polygons: Optional[Iterable[Iterable[PolygonType]]] = None,
    binary_masks: Optional[Iterable[MaskType]] = None,
    draw_box_number: bool = False,
    draw_mask_number: bool = False,
    label_mode: str = "1",
    mask_alpha: float = 0.5,
    box_thickness: int = 1,
    polygon_thickness: int = 2,
    enable_draw_box: bool = True,
    enable_draw_polygon: bool = True,
) -> np.ndarray:
    """
    Draw bbox, polygon, and mask on an image. Both input and output are RGB.

    Args:
        image: HxWx3 image, np.uint8 or PIL.Image.
        color_template: color template sequence, elements are RGB/BGR in 0-255 or 0-1 format.
                        Random colors are appended automatically when insufficient.
        boxes: optional, a list of (x0, y0, x1, y1).
        polygons: optional, structure is [List[instance polygons]]; a single polygon can be
                  flat [x0,y0,...] or Nx2.
        binary_masks: optional, list; each element is a binary mask of the same size as the image.
        draw_box_number: whether to draw a number on the bbox.
        draw_mask_number: whether to draw a number inside the mask.
        label_mode: "1" for numeric labels, "a" for a,b,c... labels.
        mask_alpha: mask fill transparency.
        box_thickness: bbox line width.
        polygon_thickness: polygon line width.
        enable_draw_box: whether to draw/generate bboxes.
        enable_draw_polygon: whether to draw/generate polygon contours.

    Returns:
        The drawn np.uint8 image (BGR).
    """
    img = _to_numpy_image(image)
    height, width = img.shape[:2]
    canvas_bgr = img[:, :, ::-1].copy()  # Convert to BGR for OpenCV drawing

    box_len = len(boxes) if boxes is not None else None
    poly_len = len(polygons) if polygons is not None else None
    mask_len = len(binary_masks) if binary_masks is not None else None
    num_instances = _validate_lengths([box_len, poly_len, mask_len])

    palette = _prepare_palette(color_template, num_instances)
    canvas = img.copy()

    norm_masks = (
        _normalize_masks(binary_masks, (height, width))
        if binary_masks is not None
        else None
    )

    norm_boxes = _normalize_boxes(boxes) if boxes is not None else None
    if norm_boxes is None and norm_masks is not None and enable_draw_box:
        norm_boxes = [_mask_to_box(m) for m in norm_masks]

    norm_polys = _normalize_polygons(polygons) if polygons is not None else None
    if norm_polys is None and norm_masks is not None and enable_draw_polygon:
        norm_polys = []
        for m in norm_masks:
            norm_polys.append(_mask_to_polygons(m))

    for idx in range(num_instances):
        color_rgb = palette[idx]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        tag = _number_to_string(idx + 1) if label_mode == "a" else str(idx + 1)

        if norm_masks is not None:
            mask = norm_masks[idx]
            mask_idx = mask > 0
            # Blend only the masked region (keeps the rest of the image unchanged)
            if mask_idx.any():
                color_arr = np.array(color_bgr, dtype=np.float32)
                canvas_masked = canvas_bgr[mask_idx].astype(np.float32)
                blended = canvas_masked * (1 - mask_alpha) + color_arr * mask_alpha
                canvas_bgr[mask_idx] = blended.astype(np.uint8)

            if draw_mask_number:
                _draw_mask_number(canvas_bgr, mask, tag, color_bgr)

        if norm_polys is not None and enable_draw_polygon:
            for poly in norm_polys[idx]:
                cv2.polylines(
                    canvas_bgr, [poly], isClosed=True, color=color_bgr, thickness=polygon_thickness
                )

        if norm_boxes is not None and enable_draw_box:
            box = norm_boxes[idx]
            cv2.rectangle(
                canvas_bgr, (box[0], box[1]), (box[2], box[3]), color_bgr, box_thickness
            )
            if draw_box_number:
                _draw_box_number(canvas_bgr, box, tag, color_bgr)

    # Convert back to RGB output
    return canvas_bgr[:, :, ::-1]


def _to_target_size(mask, target_size_hw):
    """Nearest-neighbor resize a [H,W] tensor mask to target (H, W)."""
    if isinstance(mask, torch.Tensor):
        return torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=target_size_hw,
            mode='nearest'
        ).squeeze().long()
    return mask


def create_visualization(
    ref_img,
    ref_mask,
    query_img,
    query_gt_mask,
    query_pred_mask,
    save_path=None,
    alpha=0.5,
    gap=10,
):
    """Overlay ref/gt/pred masks with fixed semantic colors and concatenate horizontally.

    Layout (left -> right): ref | query+gt | query+pred (one panel per pred).
    Fixed colors: ref=green, gt=red, pred=blue. Masks are binarized (>0) and
    drawn as a solid fill (no box, no polygon).

    Args:
        ref_img: PIL.Image of the reference image.
        ref_mask: [H,W] tensor mask (any foreground value > 0).
        query_img: PIL.Image of the query image.
        query_gt_mask: [H,W] tensor mask (ground truth).
        query_pred_mask: a single [H,W] mask or a list of them (predictions).
        save_path: if given, save the concatenated image here.
        alpha: mask fill transparency passed to draw_instances_viz.
        gap: pixel gap between panels.

    Returns:
        PIL.Image of the concatenated visualization.
    """
    REF_COLOR = (0, 255, 0)      # green
    GT_COLOR = (255, 0, 0)       # red
    PRED_COLOR = (0, 0, 255)     # blue

    if not isinstance(query_pred_mask, list):
        query_pred_mask = [query_pred_mask]

    target_size = ref_img.size          # (W, H)
    target_size_hw = target_size[::-1]  # (H, W)
    query_img = query_img.resize(target_size, Image.Resampling.LANCZOS)

    ref_mask = _to_target_size(ref_mask, target_size_hw)
    query_gt_mask = _to_target_size(query_gt_mask, target_size_hw)
    query_pred_mask = [_to_target_size(m, target_size_hw) for m in query_pred_mask]

    # Each panel: one mask overlaid on the corresponding image (solid fill only).
    ref_np = draw_instances_viz(
        image=ref_img,
        color_template=[REF_COLOR],
        binary_masks=[ref_mask],
        mask_alpha=alpha,
        enable_draw_box=False,
        enable_draw_polygon=False,
    )
    gt_np = draw_instances_viz(
        image=query_img,
        color_template=[GT_COLOR],
        binary_masks=[query_gt_mask],
        mask_alpha=alpha,
        enable_draw_box=False,
        enable_draw_polygon=False,
    )
    pred_panels = [
        draw_instances_viz(
            image=query_img,
            color_template=[PRED_COLOR],
            binary_masks=[m],
            mask_alpha=alpha,
            enable_draw_box=False,
            enable_draw_polygon=False,
        )
        for m in query_pred_mask
    ]

    img_width, img_height = target_size
    num_preds = len(pred_panels)
    total_width = img_width * (2 + num_preds) + gap * (1 + num_preds)
    result_img = Image.new('RGB', (total_width, img_height), (255, 255, 255))

    result_img.paste(Image.fromarray(ref_np, mode='RGB'), (0, 0))
    result_img.paste(Image.fromarray(gt_np, mode='RGB'), (img_width + gap, 0))
    current_x = img_width * 2 + gap * 2
    for pred_np in pred_panels:
        result_img.paste(Image.fromarray(pred_np, mode='RGB'), (current_x, 0))
        current_x += img_width + gap

    if save_path:
        result_img.save(save_path)
    return result_img


__all__ = ["draw_instances_viz", "create_visualization"]
