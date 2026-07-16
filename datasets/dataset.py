r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms

from .coco import DatasetCOCO
from .pascal import DatasetPASCAL
from .fss import DatasetFSS
from .pascal_part import DatasetPASCALPart
from .lvis import DatasetLVIS
from .isic import DatasetISIC
from .isaid import DatasetISAID

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, seed, num_test_samples=0):

        cls.num_test_samples = num_test_samples
        cls.datasets = {
            'coco': DatasetCOCO,
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'pascal_part': DatasetPASCALPart,
            'lvis': DatasetLVIS,
            'isic': DatasetISIC,
            'isaid': DatasetISAID,
        }

        cls.seed =seed
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize, seed=cls.seed, num_test_samples=cls.num_test_samples)

        return dataset
