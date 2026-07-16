import torch
from PIL import Image
from typing import Tuple
from sam3.sam3.model.box_ops import masks_to_boxes
from utils.new_viz import draw_instances_viz


def make_ref_image_for_llm(
    ref_image: torch.Tensor,
    ref_mask: torch.Tensor,
    alpha: float = 0.5,
    crop: bool = False,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Generate a visualization reference image for the LLM to extract concepts,
    based on ref_image and ref_mask:
    1) highlight the mask in red
    2) draw a green bbox around each mask

    Args:
        ref_image: image tensor of shape [3, H, W], values 0~1 or 0~255 are both fine.
        ref_mask:  binary mask of shape [N, H, W] (can be bool / 0-1 float / 0-255 uint8)
        alpha:     overlay transparency for the mask (0~1, larger means more solid red)
        crop:      if True, crop the image by the largest-area bbox and resize back
                   to the original size.

    Returns:
        vis_pil:  the processed PIL.Image
        boxes:    tensor of shape [N, 4] in xyxy format (from masks_to_boxes)
    """
    # ---------- Basic checks & preprocessing ----------
    assert ref_image.ndim == 3 and ref_image.shape[0] == 3, \
        f"ref_image shape should be [3, H, W], but got {ref_image.shape}"

    if ref_mask.ndim == 2:
        # Support the N=1 case without a batch dimension
        ref_mask = ref_mask.unsqueeze(0)

    assert ref_mask.ndim == 3, \
        f"ref_mask shape should be [N, H, W], but got {ref_mask.shape}"

    _, H, W = ref_image.shape
    N, Hm, Wm = ref_mask.shape
    assert (H, W) == (Hm, Wm), "ref_image and ref_mask spatial sizes do not match"

    # Move the image to CPU and ensure it is uint8 [0,255]
    img = ref_image.detach().cpu()
    if img.dtype.is_floating_point:
        # Assume 0~1 or 0~255, unify to 0~255
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.clamp(0, 255).byte()
    else:
        img = img.clamp(0, 255).byte()

    # [3,H,W] -> [H,W,3] -> PIL.Image
    img_np = img.permute(1, 2, 0).numpy()
    base_pil = Image.fromarray(img_np, mode="RGB")

    # ---------- Merge all masks and highlight in red ----------
    # Convert to a bool mask: [N,H,W] -> [H,W]
    if ref_mask.dtype == torch.bool:
        mask_bool = ref_mask
    else:
        mask_bool = ref_mask > 0  # support 0/1 or 0/255

    any_mask = mask_bool.any(dim=0)  # [H,W]
    has_foreground = bool(any_mask.any())

    if N > 0:
        boxes = masks_to_boxes(mask_bool.to(ref_image.device))  # [N,4] xyxy
    else:
        boxes = torch.zeros((0, 4), device=ref_image.device)

    crop_box = None
    if crop and has_foreground and boxes.shape[0] > 0:
        # Use only the largest-area bbox as the crop reference to avoid over-zooming on multiple instances
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        max_idx = int(torch.argmax(areas).item())
        x0, y0, x1, y1 = boxes[max_idx].detach().cpu().tolist()
        box_w = x1 - x0
        box_h = y1 - y0
        if box_w > 0 and box_h > 0:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            scale = 1.3
            new_w = box_w * scale
            new_h = box_h * scale
            x0 = max(0, int(round(cx - new_w / 2.0)))
            x1 = min(W, int(round(cx + new_w / 2.0)))
            y0 = max(0, int(round(cy - new_h / 2.0)))
            y1 = min(H, int(round(cy + new_h / 2.0)))
        x1 = max(x1, x0 + 1)
        y1 = max(y1, y0 + 1)
        crop_box = (x0, y0, x1, y1)

    if has_foreground:
        # Use the new visualization: draw only the mask and its contour (auto polygon), no bbox
        mask_list = [m for m in mask_bool]  # pass as list to preserve instance order
        vis_np = draw_instances_viz(
            image=img_np,
            color_template=[(255, 0, 0)] * len(mask_list),  # all masks in red
            binary_masks=mask_list,
            mask_alpha=alpha,
            enable_draw_box=False,
            enable_draw_polygon=True,
            polygon_thickness=1,
        )
        # draw_instances_viz returns RGB, can be converted to PIL directly
        vis_pil = Image.fromarray(vis_np, mode="RGB")
    else:
        # No mask, just use the original image
        vis_pil = base_pil

    if crop and crop_box is not None:
        x0, y0, x1, y1 = crop_box
        crop_w = max(1, x1 - x0)
        crop_h = max(1, y1 - y0)
        cropped = vis_pil.crop((x0, y0, x1, y1))
        # Scale proportionally, aligning the long side to the original size (no padding)
        if crop_w >= crop_h:
            scale = W / float(crop_w)
            new_w = W
            new_h = max(1, int(round(crop_h * scale)))
        else:
            scale = H / float(crop_h)
            new_h = H
            new_w = max(1, int(round(crop_w * scale)))

        vis_pil = cropped.resize((new_w, new_h), Image.BILINEAR)

        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x0) * scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y0) * scale
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

    return vis_pil, boxes
