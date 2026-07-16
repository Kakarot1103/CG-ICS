import torch

def calculate_iou(prediction, mask):
    if mask.sum() == 0:
        if prediction.sum() == 0:
            return torch.tensor(1)
        else:
            return torch.tensor(0)
    intersection = prediction * mask
    union = prediction + mask - intersection
    return (intersection.sum() / (union.sum())).item()

def compute_iou(pred_mask,ori_gt_mask,gt_name,gt_class_id,dataset,org_size):
    if 'pascal_part' in dataset.benchmark or 'pascal' not in dataset.benchmark:
        gt_cmask = ori_gt_mask
    else:
        gt_cmask = dataset.read_mask(gt_name, gt_class_id)
    if dataset.benchmark == 'pascal':
        gt_mask, query_ignore_idx = dataset.extract_ignore_idx(gt_cmask.float(), gt_class_id)
    else:
        gt_mask = gt_cmask
        query_ignore_idx = None

    pred_mask = torch.nn.functional.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0).float(),
            size=org_size,
            mode='nearest'
        ).squeeze()
    gt_mask = torch.nn.functional.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=org_size,
            mode='nearest'
        ).squeeze()
    
    iou = classify_prediction(
            pred_mask=pred_mask.float().unsqueeze(0).clone().cpu(),
            gt_mask=gt_mask.float().unsqueeze(0).clone().cpu(),
            query_ignore_idx=None if query_ignore_idx==None else query_ignore_idx.unsqueeze(0).clone()
        )
    return iou * 100
    
def classify_prediction(pred_mask, gt_mask,query_ignore_idx):
    if query_ignore_idx is not None:
        assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
        query_ignore_idx *= 255
        gt_mask = gt_mask + query_ignore_idx
        pred_mask[gt_mask == 255] = 255
    # compute intersection and union of each episode in a batch
    area_inter, area_pred, area_gt = [],  [], []
    for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
        _inter = _pred_mask[_pred_mask == _gt_mask]
        if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
            _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
        else:
            _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
        area_inter.append(_area_inter)
        area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
        area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
    area_inter = torch.stack(area_inter).t()
    area_pred = torch.stack(area_pred).t()
    area_gt = torch.stack(area_gt).t()
    area_union = area_pred + area_gt - area_inter

    return area_inter[1] / area_union[1] if area_union[1] > 0 else torch.tensor(0.0)