"""
Tree Search for Prompt Selection

TreeSearcher runs an independent tree-search (initial -> synonym expansion ->
restart -> early-stop/best) on each reference image to obtain that ref's final
concept {T*}_n; finally it takes the nshot final concepts and re-ranks them
across all reference images to select the globally optimal prompt.

Single-ref search flow (unified node loop, starting from a virtual root):
1. initial: extract N candidates from this ref -> score -> depth=0 nodes
2. early stop: a node whose combined score >= early_stop_threshold is returned immediately
3. all failed -> restart: switch direction and regenerate (up to max_restarts times)
4. tree expansion (child): generate synonyms for qualifying nodes -> score -> depth+1 nodes -> recurse
5. the single-ref final pick is the highest-scoring node across all nodes

Deduplication: each ref maintains an independent vocabulary.
Driving score = (ref_iou ** alpha) * (query_score ** beta).
"""

from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
import random
import time
import torchvision
from utils.connected import split_connected_components
from utils.ref_image_edit import make_ref_image_for_llm
from utils.metrics import calculate_iou


@dataclass
class TreeNode:
    """A node in the search tree."""
    prompt: str
    score: float
    ref_iou: float
    query_score: float
    depth: int = 0
    mode: str = 'child'                      # which generation mode produced this node: initial/child/restart
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)

    def __repr__(self):
        return f"TreeNode('{self.prompt}', score={self.score:.4f}, depth={self.depth}, mode={self.mode})"


@dataclass
class TreeSearchResult:
    """Tree search result."""
    best_prompt: str
    best_score: float
    all_nodes: List[TreeNode]
    total_evaluations: int
    num_restarts: int
    max_depth_reached: int
    vocabulary: Set[str]
    tree_search_time: float = 0.0
    num_nodes: int = 0


class TreeSearcher:
    """
    Tree-search prompt selector. Holds two model instances: segmenter (SAM3) and
    mllm (MLLM). Hyperparameters are reused from the main program's args.

    The single entry point is select(ref_imgs, ref_masks, query_img): it performs
    candidate generation + tree search internally and returns the best prompt
    (str); detailed statistics (VLM/SAM3 call counts, timing) are stored in
    self.last_result.

    Args:
        segmenter: a Sam3Segmenter instance, used for candidate scoring
        mllm: an MLLM instance, used for MLLM calls (candidate generation/expansion/restart)
        args: the main program's argparse object, must contain BON/num_loops/
              num_expand_per_node/alpha/beta/expansion_threshold/max_restarts/
              early_stop_threshold
        verbose: whether to print detailed progress
    """

    def __init__(self, segmenter, mllm, args, verbose: bool = False):
        self.segmenter = segmenter
        self.mllm = mllm
        self.BON = args.BON
        self.num_loops = args.num_loops
        self.num_expand_per_node = args.num_expand_per_node
        self.alpha = args.alpha
        self.beta = args.beta
        self.expansion_threshold = args.expansion_threshold
        self.max_restarts = args.max_restarts
        self.early_stop_threshold = args.early_stop_threshold
        self.verbose = verbose
        self.last_result: TreeSearchResult = None

    def select(self, ref_imgs: List, ref_masks: List, query_img) -> str:
        """
        Select the best prompt from reference images + query image (the single
        external entry point).

        Flow (aligned with the paper's guidelines): run an independent tree-search
        on each reference image to obtain that ref's final concept {T*}_n; then
        re-rank these nshot final concepts across all reference images to pick the
        global best.

        Args:
            ref_imgs: list of reference image tensors [tensor(3,H,W), ...]
            ref_masks: list of reference mask tensors [tensor(H,W), ...]
            query_img: query image tensor tensor(3,H,W)

        Returns:
            The best prompt (str). See self.last_result (TreeSearchResult) for
            detailed statistics.
        """
        ts_start = time.time()
        query_img_pil = torchvision.transforms.functional.to_pil_image(query_img)

        ref_imgs_pil = []
        ref_imgs_masked_pil = []
        # Per-ref preprocessing: original PIL image + red-highlighted mask image (for the MLLM)
        for ref_img, ref_mask in zip(ref_imgs, ref_masks):
            ref_img_pil = torchvision.transforms.functional.to_pil_image(ref_img)
            ref_imgs_pil.append(ref_img_pil)
            ref_ins_mask = split_connected_components(ref_mask)
            ref_img_pre, _ = make_ref_image_for_llm(ref_img, ref_ins_mask, crop=True, alpha=0.3)
            ref_imgs_masked_pil.append(ref_img_pre)

        # ========== Run an independent tree-search per ref to get that ref's final concept ==========
        finals: List[str] = []           # {T*}_n, one final concept per ref
        per_ref_results: List[TreeSearchResult] = []
        for ref_idx in range(len(ref_imgs)):
            best_prompt_i, result_i = self._search_single_ref(
                ref_idx=ref_idx,
                ref_imgs_pil=ref_imgs_pil,
                ref_masks=ref_masks,
                ref_imgs_masked_pil=ref_imgs_masked_pil,
                query_img_pil=query_img_pil,
            )
            finals.append(best_prompt_i)
            per_ref_results.append(result_i)

        # ========== Cross-reference re-rank: re-score these nshot final concepts on all refs ==========
        best_prompt, best_score = self._rerank(
            finals, ref_imgs_pil, ref_masks, query_img_pil
        )

        # When the best score is too low, fall back to pure visual (ported from the original Sam3Segmenter.select_prompt)
        if best_score < 1e-6:
            best_prompt = 'visual'

        # Aggregate the statistics from the nshot single-ref searches to keep the
        # external statistical semantics continuous
        result = self._merge_results(
            best_prompt=best_prompt,
            best_score=best_score,
            per_ref_results=per_ref_results,
            tree_search_time=time.time() - ts_start,
        )
        self.last_result = result
        return best_prompt

    def _search_single_ref(
        self,
        ref_idx: int,
        ref_imgs_pil: List,
        ref_masks: List,
        ref_imgs_masked_pil: List,
        query_img_pil,
    ) -> Tuple[str, TreeSearchResult]:
        """
        Run a full tree-search on a single reference image ref_idx (unified node
        loop, starting from a virtual root).

        Returns:
            (this ref's final concept, this ref's TreeSearchResult)
        """
        ts_start = time.time()

        # This ref's resources
        ref_img_pil = ref_imgs_pil[ref_idx]
        ref_img_masked = ref_imgs_masked_pil[ref_idx]
        ref_mask = ref_masks[ref_idx]

        # Hyperparameters
        num_loops = self.num_loops
        num_expand_per_node = self.num_expand_per_node
        alpha = self.alpha
        beta = self.beta
        expansion_threshold = self.expansion_threshold
        max_restarts = self.max_restarts
        early_stop_threshold = self.early_stop_threshold
        verbose = self.verbose

        # Per-ref vocabulary (independent dedup per ref)
        vocabulary: Set[str] = set()
        all_nodes: List[TreeNode] = []
        total_evaluations = 0
        num_restarts = 0
        max_depth_reached = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"[ref {ref_idx}] Single-ref Tree Search")
            print(f"Max depth: {num_loops}, Expansion threshold: {expansion_threshold:.2f}, "
                  f"Max restarts: {max_restarts}")
            print(f"{'='*60}\n")

        # ===== Unified node handling: candidates -> single-ref scoring -> build TreeNode (eliminates three duplicates) =====
        def build_nodes(candidates: List[str], depth: int, mode: str,
                        parent: Optional[TreeNode] = None) -> Tuple[List[TreeNode], bool]:
            """Score a batch of candidates (single ref) + build nodes; returns (node list, whether early-stop triggered)."""
            nonlocal total_evaluations
            scored = self._score_single_ref_raw(
                ref_idx, ref_mask, candidates, alpha, beta
            )
            total_evaluations += len(candidates)
            nodes: List[TreeNode] = []
            early_stop = False
            for s in scored:
                node = TreeNode(
                    prompt=s['prompt'],
                    score=s['combined_score'],
                    ref_iou=s['ref_iou'],
                    query_score=s['query_score'],
                    depth=depth,
                    mode=mode,
                    parent=parent,
                )
                nodes.append(node)
                all_nodes.append(node)
                if parent is not None:
                    parent.children.append(node)
                if node.score >= early_stop_threshold:
                    early_stop = True
            return nodes, early_stop

        # ===== Phase A: initial (virtual root) -> depth=0 nodes =====
        initial_cands = self.mllm.generate(
            mode='initial', ref_image=ref_img_pil, ref_image_masked=ref_img_masked,
            num_candidates=self.BON,
        )
        initial_cands = [c for c in initial_cands if c not in vocabulary]
        vocabulary.update(initial_cands)

        root_nodes, early = build_nodes(initial_cands, depth=0, mode='initial')
        if verbose and root_nodes:
            print(f"[ref {ref_idx}] Initial candidates scored:")
            for n in sorted(root_nodes, key=lambda x: x.score, reverse=True):
                print(f"  {n.prompt}: {n.score:.4f} "
                      f"(ref_iou={n.ref_iou:.3f}, query={n.query_score:.3f})")

        # Early stop (initial layer)
        if early:
            best = max(root_nodes, key=lambda x: x.score)
            if verbose:
                print(f"\n🎯 [ref {ref_idx}] Early stop at initial: \"{best.prompt}\" "
                      f"score={best.score:.4f}")
            return self._pack_single_ref_result(
                best.prompt, best.score, all_nodes, total_evaluations,
                num_restarts, max_depth_reached, vocabulary, ts_start)

        # ===== Phase B: all failed -> restart (up to max_restarts times) =====
        all_failed = all(n.score < expansion_threshold for n in root_nodes)
        restart_attempt = 0
        while all_failed and restart_attempt < max_restarts:
            restart_attempt += 1
            num_restarts += 1
            if verbose:
                print(f"\n⚠️  [ref {ref_idx}] All initial failed -> "
                      f"Restart {restart_attempt}/{max_restarts}")
            new_cands = self.mllm.generate(
                mode='restart', ref_image=ref_img_pil, ref_image_masked=ref_img_masked,
                num_candidates=self.BON, existing_vocab=vocabulary,
                failed_candidates=[n.prompt for n in root_nodes],
            )
            if not new_cands:
                if verbose:
                    print("  No new candidates, abort restart.")
                break
            new_cands = [c for c in new_cands if c not in vocabulary]
            vocabulary.update(new_cands)

            root_nodes, early = build_nodes(new_cands, depth=0, mode='restart')
            if verbose and root_nodes:
                print(f"[ref {ref_idx}] Restart candidates scored:")
                for n in sorted(root_nodes, key=lambda x: x.score, reverse=True):
                    print(f"  {n.prompt}: {n.score:.4f}")
            if early:
                best = max(root_nodes, key=lambda x: x.score)
                if verbose:
                    print(f"\n🎯 [ref {ref_idx}] Early stop at restart: \"{best.prompt}\" "
                          f"score={best.score:.4f}")
                return self._pack_single_ref_result(
                    best.prompt, best.score, all_nodes, total_evaluations,
                    num_restarts, max_depth_reached, vocabulary, ts_start)
            all_failed = all(n.score < expansion_threshold for n in root_nodes)

        # Restarts exhausted and still all failed -> take the best among all current nodes
        if all_failed:
            best = max(all_nodes, key=lambda x: x.score)
            if verbose:
                print(f"\n❌ [ref {ref_idx}] Still all failed after {max_restarts} restarts. "
                      f"Use best: \"{best.prompt}\" {best.score:.4f}")
            return self._pack_single_ref_result(
                best.prompt, best.score, all_nodes, total_evaluations,
                num_restarts, max_depth_reached, vocabulary, ts_start)

        # ===== Phase C: tree expansion BFS (child mode) =====
        frontier = [n for n in root_nodes if n.score >= expansion_threshold]
        if verbose and frontier:
            print(f"\n[ref {ref_idx}] Tree expansion: {len(frontier)} nodes above threshold")

        while frontier and max_depth_reached < num_loops:
            current = frontier.pop(0)
            if current.depth >= num_loops:
                continue
            if verbose:
                print(f"\n  Expanding depth {current.depth}: \"{current.prompt}\" "
                      f"(score={current.score:.4f})")

            child_cands = self.mllm.generate(
                mode='child', ref_image=ref_img_pil, ref_image_masked=ref_img_masked,
                num_candidates=num_expand_per_node,
                base_prompt=current.prompt, existing_vocab=vocabulary,
            )
            if not child_cands:
                if verbose:
                    print("    No new candidates, skip.")
                continue
            child_cands = [c for c in child_cands if c not in vocabulary]
            vocabulary.update(child_cands)
            if not child_cands:
                if verbose:
                    print("    All duplicates, skip.")
                continue
            if verbose:
                print(f"    Generated {len(child_cands)} new: {child_cands}")

            child_nodes, early = build_nodes(
                child_cands, depth=current.depth + 1, mode='child', parent=current
            )
            for n in child_nodes:
                if n.depth > max_depth_reached:
                    max_depth_reached = n.depth
                if n.score >= expansion_threshold and n.depth < num_loops:
                    frontier.append(n)
            if verbose:
                for n in sorted(child_nodes, key=lambda x: x.score, reverse=True):
                    print(f"    {n.prompt}: {n.score:.4f} "
                          f"(ref_iou={n.ref_iou:.3f}, query={n.query_score:.3f})")
            if early:
                best = max(child_nodes, key=lambda x: x.score)
                if verbose:
                    print(f"\n🎯 [ref {ref_idx}] Early stop at expansion: "
                          f"\"{best.prompt}\" score={best.score:.4f}")
                return self._pack_single_ref_result(
                    best.prompt, best.score, all_nodes, total_evaluations,
                    num_restarts, max_depth_reached, vocabulary, ts_start)

        # ===== Phase D: single-ref final selection =====
        best = max(all_nodes, key=lambda x: x.score)
        if verbose:
            print(f"\n[ref {ref_idx}] Final best: \"{best.prompt}\" {best.score:.4f} "
                  f"(nodes={len(all_nodes)}, restarts={num_restarts}, "
                  f"max_depth={max_depth_reached})")
        return self._pack_single_ref_result(
            best.prompt, best.score, all_nodes, total_evaluations,
            num_restarts, max_depth_reached, vocabulary, ts_start)

    def _pack_single_ref_result(
        self, best_prompt, best_score, all_nodes, total_evaluations,
        num_restarts, max_depth_reached, vocabulary, ts_start,
    ) -> Tuple[str, TreeSearchResult]:
        """Pack the single-ref search result into (prompt, TreeSearchResult)."""
        result = TreeSearchResult(
            best_prompt=best_prompt,
            best_score=best_score,
            all_nodes=all_nodes,
            total_evaluations=total_evaluations,
            num_restarts=num_restarts,
            max_depth_reached=max_depth_reached,
            vocabulary=vocabulary,
            tree_search_time=time.time() - ts_start,
            num_nodes=len(all_nodes),
        )
        return best_prompt, result

    def _rerank(
        self,
        finals: List[str],
        ref_imgs_pil: List,
        ref_masks: List,
        query_img_pil,
    ) -> Tuple[str, float]:
        """
        Cross-reference re-rank: re-score each ref's final concept {T*}_n on all
        reference images and pick the concept with the highest avg_combined_score
        (measures cross-reference consistency).

        Returns:
            (best concept, its re-rank score)
        """
        if not finals:
            return 'visual', 0.0
        aggregated = self.score_and_aggregate(
            ref_imgs_pil, ref_masks, query_img_pil, finals, self.alpha, self.beta
        )
        best = max(aggregated, key=lambda x: x['avg_combined_score'])
        return best['prompt'], float(best['avg_combined_score'])

    def _merge_results(
        self,
        best_prompt: str,
        best_score: float,
        per_ref_results: List[TreeSearchResult],
        tree_search_time: float,
    ) -> TreeSearchResult:
        """Aggregate the statistics from nshot single-ref searches into one external TreeSearchResult."""
        all_nodes: List[TreeNode] = []
        vocabulary: Set[str] = set()
        total_evaluations = 0
        num_restarts = 0
        max_depth_reached = 0
        for r in per_ref_results:
            all_nodes.extend(r.all_nodes)
            vocabulary |= r.vocabulary
            total_evaluations += r.total_evaluations
            num_restarts += r.num_restarts
            max_depth_reached = max(max_depth_reached, r.max_depth_reached)
        return TreeSearchResult(
            best_prompt=best_prompt,
            best_score=best_score,
            all_nodes=all_nodes,
            total_evaluations=total_evaluations,
            num_restarts=num_restarts,
            max_depth_reached=max_depth_reached,
            vocabulary=vocabulary,
            tree_search_time=tree_search_time,
            num_nodes=len(all_nodes),
        )

    def score_and_aggregate(
        self,
        ref_imgs: List,
        ref_masks: List,
        query_img_pil,
        candidates: List[str],
        alpha: float,
        beta: float,
    ) -> List[Dict[str, Any]]:
        """
        Score the candidate list on [all reference images] and aggregate across
        references (used for the final re-rank). Reuses the states already built
        on self.segmenter (ref_states / query_state).

        Scoring logic:
            - ref_iou: IoU of each candidate prompt on each ref (vs that ref_mask)
            - query_score: presence_score of each candidate prompt on the query
            - combined: random when alpha=beta=0; otherwise (ref_iou ** alpha) * (query_score ** beta)
          Across references, the same prompt is averaged (avg_ref_iou / avg_query_score / avg_combined_score).

        Returns:
            List of aggregated scoring dicts.
        """
        loop_scores = []
        for ref_idx, (ref_img, ref_mask) in enumerate(zip(ref_imgs, ref_masks)):
            loop_scores.extend(
                self._score_single_ref_raw(ref_idx, ref_mask, candidates, alpha, beta)
            )
        return aggregate_scores(loop_scores)

    def _score_single_ref_raw(
        self,
        ref_idx: int,
        ref_mask,
        candidates: List[str],
        alpha: float,
        beta: float,
    ) -> List[Dict[str, Any]]:
        """Internal implementation of single-ref scoring; returns a list of raw per-candidate scoring records."""
        ref_state = self.segmenter.ref_states[ref_idx]

        # Binarize ref_mask (aligned with the original select_prompt handling)
        ref_mask_tensor = ref_mask.float()
        while ref_mask_tensor.ndim > 2:
            ref_mask_tensor = ref_mask_tensor.squeeze(0)
        ref_mask_binary = (ref_mask_tensor > 0.5).float().cpu()

        out: List[Dict[str, Any]] = []
        for prompt in candidates:
            # Compute IoU on the ref
            res_ref = self.segmenter.infer(ref_state, prompt=prompt)
            pred = res_ref['semantic_mask']
            pred_binary = (pred > 0.5).float().cpu()
            ref_iou_value = float(calculate_iou(pred_binary, ref_mask_binary))

            # Take presence_score on the query
            res_q = self.segmenter.infer(self.segmenter.query_state, prompt=prompt)
            query_score_value = float(res_q['presence_score'])

            # combined (original select_prompt logic)
            if alpha == 0 and beta == 0:
                combined_score = random.random()
            else:
                combined_score = (ref_iou_value ** alpha) * (query_score_value ** beta)

            out.append({
                "prompt": prompt,
                "ref_iou": ref_iou_value,
                "query_score": query_score_value,
                "combined_score": float(combined_score),
            })
        return out


def aggregate_scores(scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate scores of the same prompt across multiple reference images.

    Args:
        scores: list of scoring dicts, each containing prompt/ref_iou/query_score/combined_score

    Returns:
        List of aggregated scoring dicts, containing avg_ref_iou/avg_query_score/avg_combined_score/count
    """
    prompt_groups = {}
    for item in scores:
        p = item['prompt']
        if p not in prompt_groups:
            prompt_groups[p] = {
                'prompt': p,
                'ref_iou_sum': 0.0,
                'query_score_sum': 0.0,
                'combined_score_sum': 0.0,
                'count': 0
            }

        prompt_groups[p]['ref_iou_sum'] += item['ref_iou']
        prompt_groups[p]['query_score_sum'] += item['query_score']
        prompt_groups[p]['combined_score_sum'] += item['combined_score']
        prompt_groups[p]['count'] += 1

    aggregated = []
    for p, data in prompt_groups.items():
        cnt = data['count']
        aggregated.append({
            'prompt': p,
            'avg_ref_iou': data['ref_iou_sum'] / cnt,
            'avg_query_score': data['query_score_sum'] / cnt,
            'avg_combined_score': data['combined_score_sum'] / cnt,
            'count': cnt
        })

    return aggregated
