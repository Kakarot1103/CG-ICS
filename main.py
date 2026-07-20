import os
import random
import time
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torchvision

from PIL import Image
from tqdm import tqdm

from datasets.dataset import FSSDataset

from model.sam3 import Sam3Segmenter
from model.mllm import MLLM
from model.tree_search import TreeSearcher

import swanlab as wandb
from utils.logger import setup_logger
from utils.metrics import compute_iou

from utils.connected import split_connected_components
from utils.new_viz import create_visualization
from utils.ref_image_edit import make_ref_image_for_llm


# ----------------- Utilities: seed & DDP initialization -----------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_mode(args):
    """
    When launched with torchrun, read RANK / WORLD_SIZE / LOCAL_RANK from env vars.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True

        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
    else:
        # Single-GPU / non-distributed run
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False


# ----------------- Model & dataset preparation -----------------

def prepare_for_eval(args, device, rank):
    """
    Initialize FSSDataset, Sam3Segmenter, MLLM, TreeSearcher.
    """
    FSSDataset.initialize(
        img_size=args.img_size,
        datapath=args.datapath,
        use_original_imgsize=False,
        seed=args.seed,
        num_test_samples=args.num_test_samples
    )
    dataset = FSSDataset.build_dataloader(
        args.benchmark,
        args.bsz,
        args.nworker,
        args.fold,
        'test',
        args.nshot
    )

    segmenter = Sam3Segmenter(checkpoint_path=args.sam3_ckpt).to(device)
    segmenter.eval()

    mllm = MLLM(
        api_key='EMPTY',
        base_url='http://localhost:22002/v1',
        model_name='qwen3-vl-4b',
        seg_type='part' if 'part' in dataset.benchmark else 'object',
    )
    tree_searcher = TreeSearcher(
        segmenter=segmenter,
        mllm=mllm,
        args=args,
        verbose=(rank == 0 and args.save_image),
    )

    return segmenter, dataset, tree_searcher


# ----------------- Evaluation logic (DDP) -----------------

def evalute(args, segmenter, dataset, tree_searcher, device, rank, world_size):
    is_distributed = args.distributed
    is_main_process = (rank == 0)

    # DataLoader + DistributedSampler
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = None

    train_loader = DataLoader(
        dataset,
        batch_size=args.bsz,
        num_workers=args.nworker,
        sampler=sampler,
        shuffle=False if sampler is not None else False
    )

    n_classes = len(dataset.class_ids)

    # Local statistics (accumulated per-process)
    class_ious = torch.zeros(n_classes, device=device)
    class_num = torch.zeros(n_classes, device=device)

    # Latency statistics
    total_samples = 0
    total_time = 0.0
    total_prompt_time = 0.0

    testing_step = 0

    total_batches = len(train_loader)

    if is_main_process:
        progress_bar = tqdm(
            total=total_batches,
            desc=f'Testing (rank {rank})',
            unit=' batch'
        )
    else:
        progress_bar = None

    # Per-rank temp directory so cap.png files are not overwritten by other ranks
    rank_tmp_dir = os.path.join(args.save_dir, f"rank_{rank}")
    os.makedirs(rank_tmp_dir, exist_ok=True)

    if sampler is not None:
        sampler.set_epoch(0)
    class_id_to_name = {}
    for batch_idx, batch in enumerate(train_loader):
        class_idx = dataset.class_ids.index(batch['class_id'].cpu().item())
        class_id_to_name[class_idx] = batch['category'][0]

        # Sample-level statistics
        sample_start = time.time()

        batch['support_img'] = batch['support_imgs'][0]
        batch['support_mask'] = batch['support_masks'][0]
        batch['support_name'] = batch['support_names'][0]

        with torch.no_grad():
            n_support = len(batch['support_img'])
            ref_imgs = [batch['support_img'][i].squeeze() for i in range(n_support)]
            ref_masks = [batch['support_mask'][i].squeeze() for i in range(n_support)]
            query_img = batch['query_img'].squeeze()

            # ============ STEP 0: build state (image encoder runs once) ============
            ref_imgs_pil = [torchvision.transforms.functional.to_pil_image(r) for r in ref_imgs]
            query_image_pil = torchvision.transforms.functional.to_pil_image(query_img)
            cats_pil = [
                torchvision.transforms.functional.to_pil_image(
                    torch.cat([ref_imgs[i], query_img], dim=2)
                ) for i in range(n_support)
            ]
            segmenter.build_state(ref_imgs_pil, query_image_pil, cats_pil)

            # ============ STEP 1+2: tree_search selects the best prompt (reuses built state) ============
            prompt = tree_searcher.select(ref_imgs, ref_masks, query_img)
            sample_prompt_time = tree_searcher.last_result.tree_search_time

            # ============ STEP 3: visual branch — derive query instance mask from cat state ============
            query_ins_masks = []
            for i in range(n_support):
                cat_mask = torch.cat([ref_masks[i], torch.zeros_like(ref_masks[i])], dim=1)
                cat_ins_mask = split_connected_components(cat_mask)
                res_cat = segmenter.infer(segmenter.cat_states[i], mask=cat_ins_mask)
                instances = res_cat['instances']  # [M,1,H,Wcat]
                # Slice the query half (cat image is horizontally concatenated, width = ref+query)
                cat_width = instances.shape[-1]
                query_ins_masks.append(instances[:, :, :, cat_width // 2:].squeeze(1).to(device))  # [M,H,W]
            query_ins_mask = torch.cat(query_ins_masks, dim=0)

            # ============ STEP 4: run prompt + bbox inference on the query to get the mask ============
            res = segmenter.infer(segmenter.query_state, prompt=prompt, mask=query_ins_mask)
            pred_mask = res['semantic_mask']

            # ============ Clear state ============
            segmenter.clear()

        org_size = (batch['org_query_imsize'][1], batch['org_query_imsize'][0])

        # End of sample-level statistics
        sample_time = time.time() - sample_start
        total_prompt_time += sample_prompt_time
        total_samples += 1
        total_time += sample_time

        # ============ STEP 5: metrics / log / visualization ============
        # IoU (per sample) — add to this process's local statistics
        iou = compute_iou(pred_mask, batch['query_mask'].squeeze(), batch['query_name'][0],
                      dataset.class_ids[class_idx], dataset, org_size)

        class_num[class_idx] += 1
        class_ious[class_idx] += iou.squeeze().to(device)

        testing_step += 1

        # ========== Per-step: temporary all_reduce to compute current global statistics ==========
        if dist.is_available() and dist.is_initialized():
            stats = torch.stack([class_ious, class_num], dim=0)  # [2, C]
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            iou_global, class_num_global = stats
        else:
            iou_global, class_num_global = class_ious, class_num

        # Direct division; use nanmean to ignore unseen classes
        per_class_iou = iou_global / class_num_global
        miou_global = torch.nanmean(per_class_iou)

        # Visualization
        if args.save_image:
            ref_img_pil = transforms.ToPILImage()(batch['support_img'][0].cpu())
            ref_mask_cpu = batch['support_mask'][0].cpu()

            query_img_pil = transforms.ToPILImage()(batch['query_img'].squeeze(0).cpu())
            query_gt_mask = batch['query_mask'].squeeze(0).cpu()

            query_pred_mask = [
                torch.nn.functional.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(512, 512),
                    mode='nearest'
                ).squeeze()
            ]

            data_class = class_idx
            class_name = batch['category'][0]
            query_name = batch['query_name'][0].replace('.png', '').replace('jpg', '').split('/')[-1]

            save_dir = os.path.join(
                args.save_dir,
                'vis',
                f'{data_class}_' + f'{class_name}',
                f"{query_name}_{prompt}_iou_{iou.item():.1f}"
            )
            os.makedirs(save_dir, exist_ok=True)

            create_visualization(
                ref_img_pil,
                ref_mask_cpu,
                query_img_pil,
                query_gt_mask,
                query_pred_mask,
                save_path=os.path.join(save_dir, 'result.png')
            )

            # Regenerate the masked reference image (candidate generation already done inside tree_search) for the concatenated visualization
            ref_ins_mask = split_connected_components(ref_mask_cpu)
            ref_img_pre, _ = make_ref_image_for_llm(batch['support_img'][0].cpu(), ref_ins_mask, crop=True, alpha=0.3)

            # Horizontally concatenate ref_img_pil, ref_img_pre and query_img_pil
            widths = [ref_img_pil.width, ref_img_pre.width, query_img_pil.width]
            heights = [ref_img_pil.height, ref_img_pre.height, query_img_pil.height]
            max_height = max(heights)
            total_width = sum(widths)

            combined_img = Image.new('RGB', (total_width, max_height))
            combined_img.paste(ref_img_pil, (0, 0))
            combined_img.paste(ref_img_pre, (widths[0], 0))
            combined_img.paste(query_img_pil, (widths[0] + widths[1], 0))
            combined_img.save(os.path.join(save_dir, f'{prompt}.png'))

        if is_main_process:
            # In per_class_iou: nan means "class never tested"; 0.0000 means "tested but IoU is 0"
            wandb.log({
                "testing_step": testing_step,
                "testing/miou": miou_global,
                "testing/num_sample": class_num_global.sum().item(),
                "testing/sample_prompt_time": sample_prompt_time,
                "testing/sample_time": sample_time,
            }, step=testing_step)

            logger.info(
                f'''[GLOBAL] Step {testing_step}/{total_batches},
                    per-class num (global): {[f'{int(x)}' for x in class_num_global.tolist()]},
                    per-class IoU (global): {[f'{float(x):.3f}' for x in per_class_iou.tolist()]},
                    mIoU (global): {miou_global:.3f},
                    [Per-Sample] PromptTime: {sample_prompt_time:.2f}s, TotalTime: {sample_time:.2f}s
'''
            )

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(miou=float(miou_global))

    if progress_bar is not None:
        progress_bar.close()

    # After the loop: final all_reduce to get the final global statistics
    if dist.is_available() and dist.is_initialized():
        stats = torch.stack([class_ious, class_num], dim=0)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        iou_global, class_num_global = stats
    else:
        iou_global, class_num_global = class_ious, class_num
    if is_main_process:
        lines = []
        for idx, cnt in enumerate(class_num_global.tolist()):
            if cnt == 0:
                continue
                # Prefer runtime-collected names; fall back to class_ids if missing
                name = class_id_to_name.get(idx, str(dataset.class_ids[idx]))
                cls_iou = float(iou_global[idx] / cnt)
                lines.append(f"{name}: {cls_iou:.3f}")
        logger.info("Per-class IoU: \n" + "\n".join(lines))
    if is_main_process:
        logger.info(f"[FINAL GLOBAL] mIoU: {miou_global:.3f}")
        # Aggregate latency statistics
        if total_samples > 0:
            avg_time = total_time / total_samples
            avg_prompt_time = total_prompt_time / total_samples
            logger.info(
                f"[STATS SUMMARY] Samples: {total_samples}, "
                f"Avg prompt selection time: {avg_prompt_time:.2f}s, "
                f"Avg total latency: {avg_time:.2f}s, "
                f"Total time: {total_time:.1f}s"
            )
            wandb.log({
                "stats/avg_prompt_time": avg_prompt_time,
                "stats/avg_latency": avg_time,
            })



# ----------------- main -----------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch DDP Inference')

    # Data_Dir
    parser.add_argument('--datapath', type=str, default='data')
    parser.add_argument('--sam3_ckpt', type=str,
                        default='Pretrained_models/sam3.pt',
                        help='Path to the SAM3 checkpoint (.pt)')
    parser.add_argument('--benchmark', type=str, default='pascal_part',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'pascal_part', 'isic', 'isaid'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num_test_samples', type=int, default=0, help='Number of test samples (0 means use default: pascal=1000, lvis=2300, pascal_part=min(len,1000))')
    parser.add_argument('--BON', type=int, default=5, help='Number of caption candidates generated for prompt selection')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for prompt selection')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for prompt selection')

    # Tree Search parameters
    parser.add_argument('--num_loops', type=int, default=3, help='Maximum tree depth (number of expansion levels)')
    parser.add_argument('--num_expand_per_node', type=int, default=5, help='Number of children to generate per node expansion (default: 5)')
    parser.add_argument('--expansion_threshold', type=float, default=0.5, help='Score threshold for node expansion (nodes >= threshold are expanded, default: 0.5)')
    parser.add_argument('--max_restarts', type=int, default=3, help='Maximum number of restart attempts when all initial candidates fail (default: 3)')
    parser.add_argument('--early_stop_threshold', type=float, default=0.8, help='Early stopping threshold - return immediately if any node reaches this score (default: 0.9, set to 1.0 to disable)')

    # Save_Dir
    parser.add_argument('--save_dir', type=str, default='./checkpoints/debug', help='Directory to save logs and visualizations')
    parser.add_argument('--save_image', type=int, default=1, help='Whether to save visualizations')

    # Testing
    parser.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()

    # Initialize DDP
    init_distributed_mode(args)
    rank = args.rank
    world_size = args.world_size
    is_main_process = (rank == 0)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda", args.local_rank if args.distributed else 0)
    else:
        device = torch.device("cpu")

    # Use different seeds per rank to avoid identical sampling
    set_seed(args.seed + rank)

    # swanlab / wandb
    os.environ["WANDB_MODE"] = 'disabled'
    wandb_name = f"{args.benchmark}_" + args.save_dir.split('/')[-1] + '_' + f"f{args.fold}_shot{args.nshot}"

    if is_main_process:
        wandb.init(project="sam3-fss", name=wandb_name, group=args.benchmark, config=args)

    args.save_dir = os.path.join(args.save_dir, wandb_name)
    os.makedirs(args.save_dir, exist_ok=True)

    global logger
    logger = setup_logger(log_file=os.path.join(args.save_dir, f'train_rank{rank}.log'))

    if is_main_process:
        logger.info('Preparing model and dataset...')
    segmenter, dataset, tree_searcher = prepare_for_eval(args, device, rank)

    evalute(args, segmenter, dataset, tree_searcher, device, rank, world_size)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
