import torch
import torch.nn as nn
from typing import List, Optional
from PIL import Image

from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
from sam3.sam3.model.box_ops import box_xyxy_to_cxcywh, masks_to_boxes
from sam3.sam3.visualization_utils import normalize_bbox


class Sam3Segmenter(nn.Module):
    """
    A pure segmentation primitive: only responsible for running segmentation
    inference on an already-built state.

    Usage:
        1. build_state(refs, query, cats)  — run the image encoder, cache ref/query/cat states
        2. infer(state, prompt, mask)      — reuse the state for segmentation (can be called repeatedly)
        3. clear()                         — release GPU memory when done

    The state is maintained by the processor: set_image runs the encoder and stores
    backbone_out; set_text_prompt / add_geometric_prompt reuse backbone_out to run the
    head; reset_all_prompts only clears prompts/results while keeping backbone_out.
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path).to(self.device)
        self.processor = Sam3Processor(self.model)

        # Cached states (written by build_state, cleared by clear)
        self.ref_states: Optional[List[dict]] = None
        self.query_state: Optional[dict] = None
        self.cat_states: Optional[List[dict]] = None

    def build_state(self, refs: List[Image.Image], query: Image.Image, cats: Optional[List[Image.Image]] = None):
        """
        Run the image encoder to build and cache states. Each image triggers one
        independent encoder call.

        Args:
            refs: list of reference PIL images (one state per ref, stored in self.ref_states)
            query: query PIL image (stored in self.query_state)
            cats: list of concatenated PIL images, same length as refs (optional; if None,
                  cat states are not built)
        """
        self.ref_states = [self.processor.set_image(ref) for ref in refs]
        self.query_state = self.processor.set_image(query)
        if cats is not None:
            self.cat_states = [self.processor.set_image(cat) for cat in cats]
        else:
            self.cat_states = None

    def clear(self):
        """Clear all cached states and release GPU memory."""
        self.ref_states = None
        self.query_state = None
        self.cat_states = None

    def infer(self, state, prompt: Optional[str] = None, mask: Optional[torch.Tensor] = None) -> dict:
        """
        Run segmentation inference on the given state (reuses the backbone_out
        built by build_state).

        Args:
            state: a state produced by build_state (self.ref_states[i] /
                   self.query_state / self.cat_states[i])
            prompt: text prompt (optional). If provided, set_text_prompt is called.
            mask: instance mask [N,H,W] (optional). If provided, it is converted to
                  normalized cxcywh bboxes and passed to add_geometric_prompt.
            If both are None, an error is raised (at least one prompt type is required).

        Mechanism (prompts are stacked serially):
            - If prompt is set: first set_text_prompt (writes language_features, runs one forward)
            - If mask is set: then add_geometric_prompt (appends a box, runs another forward)
            - When both are set, the second forward automatically becomes a text+visual
              joint inference
            - reset_all_prompts is called at the start of every invocation to prevent
              boxes from accumulating across calls

        Returns:
            {
                'semantic_mask':  [H,W] aggregated binary mask (float),
                'instances':      [M,1,H,W] instance-level binary masks,
                'presence_score': float
            }
        """
        if prompt is None and mask is None:
            raise ValueError("At least one prompt type is required (prompt or mask)")

        # Clear the previous prompts/results (backbone_out is kept)
        self.processor.reset_all_prompts(state)

        # ---- 1. text (optional) ----
        if prompt is not None:
            self.processor.set_text_prompt(prompt=prompt, state=state)

        # ---- 2. bbox (optional, converted from mask) ----
        if mask is not None:
            # Reuse the original conversion chain: masks_to_boxes(xyxy) -> cxcywh -> normalize -> tolist
            # original_width/height in state are recorded by set_image
            width = state["original_width"]
            height = state["original_height"]
            bbox_xyxy = masks_to_boxes(mask)
            norm_boxes_cxcywh = normalize_bbox(box_xyxy_to_cxcywh(bbox_xyxy).view(-1, 4), width, height).tolist()
            for box in norm_boxes_cxcywh:
                if sum(box) == 0:
                    continue
                self.processor.add_geometric_prompt(box=box, label=True, state=state)

        # ---- 3. Extract results ----
        width = state["original_width"]
        height = state["original_height"]
        if state.get('masks') is None:
            # No prediction: fall back to an empty result
            return {
                'semantic_mask': torch.zeros(height, width),
                'instances': torch.zeros(0, 1, height, width),
                'presence_score': 0.0,
            }

        instances = state['masks']  # [M,1,H,W] bool
        semantic_mask = state["semantic_mask"].float().sum(dim=0).squeeze()
        semantic_mask[semantic_mask > 1] = 1
        presence_score = float(state.get('presence_score', 0.0))

        return {
            'semantic_mask': semantic_mask,
            'instances': instances,
            'presence_score': presence_score,
        }
