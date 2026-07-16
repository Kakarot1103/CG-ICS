import torch.nn as nn
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
from typing import List, Dict

# ==================== Unified Tree-Search Prompt ====================
# The system prompt statically describes the full rules for three modes
# (initial/child/restart); on each call the user content only carries the
# "currently active mode block" (with dynamic data).
# Placeholders: {num_candidates} / {candidate_lines} are filled by MLLM.generate.

SYSTEM_PROMPT_OBJECT = (
    "You are a vision-language model specialized in understanding object-level concepts from reference images.\n"
    "Your output must be English noun phrases of 1-3 words.\n"
    "Your task is to generate concept candidates for the target object indicated by the reference mask.\n\n"
    "=== REFERENCE IMAGES ===\n"
    "You will receive reference images to help you understand the target:\n"
    "1) the original reference image with full context\n"
    "2) the same reference image with the target area highlighted in red\n"
    "Carefully examine the red-highlighted region to understand what concept you are generating alternatives for.\n\n"
    "There are three possible modes:\n\n"
    "Mode 1: Initial Concept Generation\n"
    "Generate several plausible noun phrases that describe the target object indicated by the red-highlighted region.\n\n"
    "Mode 2: Concept Expansion\n"
    "You will be given a high-performing base concept. Generate alternative noun phrases that refer to the SAME object concept.\n"
    "Examples of good synonym generation:\n"
    "  car -> automobile, vehicle, motor vehicle\n"
    "  wheel -> tire, rim, alloy wheel\n"
    "  dog -> canine, hound, pet\n"
    "Important constraints:\n"
    "- The new phrases MUST describe the SAME concept as the base.\n"
    "- Do NOT generate phrases that describe different but related concepts.\n"
    "- Focus on synonyms, alternative names, or more specific subtypes.\n"
    "- Avoid generic terms that could describe multiple different objects.\n\n"
    "Mode 3: Concept Restart\n"
    "ALL previous candidates performed poorly (scores below threshold), meaning the current concept/category direction is WRONG.\n"
    "You must ABANDON the previous direction and try a FUNDAMENTALLY DIFFERENT object category or perspective.\n"
    "What to do:\n"
    "- Try completely different object categories.\n"
    "- Consider different levels of abstraction (more general or more specific).\n"
    "- Look at the image from a different perspective.\n"
    "- Avoid repeating patterns from failed candidates.\n\n"
    "=== OUTPUT REQUIREMENTS ===\n"
    "- Output exactly {num_candidates} English noun phrase candidates.\n"
    "- Each candidate must be 1 to 3 English words.\n"
    "- Use broad, common category names or specific subtypes.\n"
    "- Do NOT output full sentences.\n"
    "- Order candidates from most likely to least likely.\n"
    "- Avoid obvious duplicates with previously used phrases.\n\n"
    "Output format (strictly):\n"
    "{candidate_lines}\n"
)

SYSTEM_PROMPT_PART = (
    "You are a vision-language model specialized in understanding part-level concepts from reference images.\n"
    "Your output must be English noun phrases of 1-3 words.\n"
    "Your goal is to name the underlying part of an object, even if the highlighted regions only cover sections or "
    "disconnected pieces of that part (e.g., part of a wheel or multiple legs). "
    "You must output the part concept (e.g., \"leg\", \"wheel\") instead of the whole-object concept (e.g., \"horse\", \"car\").\n"
    "Your task is to generate concept candidates for the target part indicated by the reference mask.\n\n"
    "=== REFERENCE IMAGES ===\n"
    "You will receive reference images to help you understand the target:\n"
    "1) the original reference image with full context\n"
    "2) a cropped reference image tightly cropped around the target part\n"
    "Carefully examine the cropped region to understand what part you are generating alternatives for.\n\n"
    "There are three possible modes:\n\n"
    "Mode 1: Initial Concept Generation\n"
    "Generate several plausible noun phrases that describe the target part indicated by the red-highlighted region.\n"
    "Use common part names (not whole-object names).\n\n"
    "Mode 2: Concept Expansion\n"
    "You will be given a high-performing base concept. Generate alternative noun phrases that refer to the SAME part concept.\n"
    "Examples of good synonym generation:\n"
    "  wheel -> tire, rim, alloy wheel\n"
    "  leg -> limb, hind leg, front leg\n"
    "  door -> door panel, car door, vehicle door\n"
    "Important constraints:\n"
    "- The new phrases MUST describe the SAME part concept as the base.\n"
    "- Do NOT generate phrases that describe different parts or whole objects.\n"
    "- Focus on synonyms, alternative names, or more specific subtypes.\n"
    "- Avoid generic terms that could describe multiple different parts.\n\n"
    "Mode 3: Concept Restart\n"
    "ALL previous candidates performed poorly (scores below threshold), meaning the current part concept direction is WRONG.\n"
    "You must ABANDON the previous direction and try a FUNDAMENTALLY DIFFERENT part or perspective.\n"
    "What to do:\n"
    "- Try completely different parts of the object.\n"
    "- Consider different levels of specificity (more general or more specific).\n"
    "- Look at the image from a different perspective.\n"
    "- Avoid repeating patterns from failed candidates.\n\n"
    "=== OUTPUT REQUIREMENTS ===\n"
    "- Output exactly {num_candidates} English noun phrase candidates.\n"
    "- Each candidate must be 1 to 3 English words.\n"
    "- Use common part names (not whole-object names).\n"
    "- Do NOT output full sentences.\n"
    "- Order candidates from most likely to least likely.\n"
    "- Avoid obvious duplicates with previously used phrases.\n\n"
    "Output format (strictly):\n"
    "{candidate_lines}\n"
)


def encode_image_to_data_url(image: Image.Image, format_: str = "PNG"):
    """Convert a PIL image into a data URL string."""
    buffered = BytesIO()
    image.save(buffered, format=format_)
    mime = f"image/{format_.lower()}"
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class MLLM(nn.Module):
    """
    An MLLM client for tree-search (OpenAI-compatible interface).

    The single generation entry point is generate(mode, ...): it generates a list
    of concept candidates according to the node mode 'initial' / 'child' / 'restart'.
    The system prompt (static) fully describes the rules for all three modes; each
    call only places the currently active mode block (with dynamic data) in the user
    content.
    """

    def __init__(
        self,
        api_key: str = "none",
        base_url: str = "http://localhost:22002/v1",
        model_name: str = "qwen3-vl-plus",
        seg_type: str = 'object',
    ):
        super().__init__()
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.api_model = model_name
        self.seg_type = seg_type

    # -------------------------
    # Utility functions
    # -------------------------

    @staticmethod
    def _parse_stage1_output(text: str, topk: int = 3) -> List[str]:
        """
        Parse the stage-1 output into [cand1, cand2, ...].
        """
        lines = text.strip().splitlines()
        cands: List[str] = []

        # Prefer parsing the "1. xxx" form
        for ln in lines:
            s = ln.strip()
            if not s or not s[0].isdigit():
                continue
            num_end = 0
            while num_end < len(s) and s[num_end].isdigit():
                num_end += 1
            if num_end < len(s) and s[num_end] in {".", ":"}:
                cands.append(s[num_end + 1:].strip())

        # If fewer than topk, fall back to scraping plain lines
        if len(cands) < topk:
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                if s[0].isdigit():
                    num_end = 0
                    while num_end < len(s) and s[num_end].isdigit():
                        num_end += 1
                    if num_end < len(s) and s[num_end] in {".", ":"}:
                        continue
                if len(s.split()) <= 4:
                    cands.append(s)
                if len(cands) >= topk:
                    break

        # Clean + pad
        cands = [c for c in cands if c]
        if len(cands) == 0:
            cands = ["object", "part", "thing"]
        cands = cands[:topk]
        if len(cands) < topk:
            cands += [cands[-1]] * (topk - len(cands))

        return cands

    

    # -------------------------
    # Low-level: API call
    # -------------------------

    def _call_api_once(
        self,
        messages: List[Dict],
        max_tokens: int = 128,
    ) -> str:
        """
        Multi-image + text call via the OpenAI-compatible (DashScope) API.
        """
        resp = self.client.chat.completions.create(
            model=self.api_model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # -------------------------
    # Single generation entry point: uniformly handles initial / child / restart node modes
    # -------------------------

    def generate(
        self,
        mode: str,
        ref_image: Image.Image,
        ref_image_masked: Image.Image = None,
        num_candidates: int = 3,
        base_prompt: str = None,
        existing_vocab: set = None,
        failed_candidates: List[str] = None,
        max_new_tokens: int = 128,
    ) -> List[str]:
        """
        Generate concept candidates according to the tree-search node mode.

        The system prompt (static) already encodes the full rules for all three
        modes; each call only places the "currently active mode block" (with dynamic
        data such as num_candidates / base_prompt / used vocabulary / failed
        candidates) in the user content.

        Args:
            mode: node mode
                - 'initial': initial node, independently extract concepts from the
                  reference image (no parent node).
                - 'child':   child node, generate synonymous/equivalent phrases based
                  on base_prompt.
                - 'restart': restart after all candidates failed, avoid
                  failed_candidates and switch direction.
            ref_image: original reference image (PIL).
            ref_image_masked: reference image with the mask highlighted (PIL), optional.
            num_candidates: number of candidates to generate.
            base_prompt: parent concept for child mode.
            existing_vocab: used vocabulary, for child/restart deduplication hints.
            failed_candidates: failed candidates to avoid in restart mode.
            max_new_tokens: maximum number of generated tokens.

        Returns:
            List of candidate concept phrases (length approximately num_candidates).
        """
        if existing_vocab is None:
            existing_vocab = set()
        if failed_candidates is None:
            failed_candidates = []

        # ---- 1. Build the active block of user content per mode (with dynamic data) ----
        if mode == 'initial':
            user_text = (
                "=== CURRENT MODE: Initial Concept Generation ===\n"
                f"Generate {num_candidates} plausible English noun phrases that describe "
                f"the target {self.seg_type} indicated by the red-highlighted region."
            )
        elif mode == 'child':
            if not base_prompt:
                raise ValueError("child mode requires base_prompt")
            user_text = (
                "=== CURRENT MODE: Concept Expansion ===\n"
                f"We have found a good concept that performs well: \"{base_prompt}\".\n"
                f"Generate {num_candidates} synonyms or semantically similar phrases that "
                f"describe the SAME {self.seg_type} concept."
            )
            if existing_vocab:
                user_text += (
                    "\nAvoid exact duplicates of already-used phrases.\n"
                    f"Already used (DO NOT repeat): {', '.join(sorted(existing_vocab))}"
                )
        elif mode == 'restart':
            user_text = (
                "=== CURRENT MODE: Concept Restart ===\n"
                "ALL previous candidates performed poorly (scores below threshold), "
                "meaning the current concept/category direction is WRONG.\n"
                f"ABANDON the previous direction and generate {num_candidates} "
                f"FUNDAMENTALLY DIFFERENT {self.seg_type} candidates."
            )
            if failed_candidates:
                user_text += "\nFailed candidates to AVOID (do NOT use similar patterns):\n"
                for cand in failed_candidates[:5]:
                    user_text += f"  - \"{cand}\"\n"
            if existing_vocab:
                user_text += (
                    "\nAvoid exact duplicates of already-used phrases.\n"
                    f"Already used (DO NOT repeat): {', '.join(sorted(existing_vocab))}"
                )
        else:
            raise ValueError(f"Unknown mode: {mode} (expected 'initial' / 'child' / 'restart')")

        # ---- 2. Build user content (image + active block) ----
        content = [
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(ref_image)},
            },
        ]
        if ref_image_masked is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(ref_image_masked)},
            })
        content.append({"type": "text", "text": user_text})

        # ---- 3. Pick the system template and format (only num_candidates/candidate_lines placeholders remain) ----
        candidate_lines = "\n".join(
            f"{idx}. candidate_{idx}" for idx in range(1, num_candidates + 1)
        )
        template = SYSTEM_PROMPT_OBJECT if self.seg_type == 'object' else SYSTEM_PROMPT_PART
        system_prompt = template.format(
            num_candidates=num_candidates,
            candidate_lines=candidate_lines,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # ---- 4. Call + parse ----
        raw_output = self._call_api_once(messages, max_tokens=max_new_tokens)
        return self._parse_stage1_output(raw_output, topk=num_candidates)
