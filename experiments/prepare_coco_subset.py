import os
import argparse
from pathlib import Path
from typing import Dict, List

import torchvision.datasets as dset

from assl_od.data import compute_statistics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for the extraction of a subset of the COCO '
                    'dataset.')

    parser.add_argument("--data_path", type=Path, required=True,
                    help="Path to COCO images folder.")
    parser.add_argument("--annotations_path", type=Path, required=True,
                    help="Path to COCO annotations .json file.")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Path to output folder, where the COCO subset "
                             "will be stored.")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                        help="Categories to be extracted from the dataset.")

    args = parser.parse_args()

    return args


def read_dataset(data_path: Path, annotations_path: Path) -> dset.CocoDetection:
    data = dset.CocoDetection(root=data_path, annFile=annotations_path)

    return data


def generate_subset(data_path: Path, annotations_path: Path, 
                    output_path: Path, categories: List[str],
                    verbose: bool = True):
    # Prepare output folders
    imgs_folder = data_path.name
    if not os.path.exists(output_path / imgs_folder):
        os.makedirs(output_path / imgs_folder)
    annotations_folder = annotations_path.parent.name
    annotations_filename = annotations_path.name
    if not os.path.exists(output_path / annotations_folder):
        os.makedirs(output_path / annotations_folder)

    # Read original COCO dataset
    data = read_dataset(data_path=data_path, 
                        annotations_path=annotations_path)

    # if verbose:
    #     # Display statistics for the newly created dataset
    #     dataset_stats = compute_statistics(data)
    #     print(f"Dataset statistics:")
    #     for key, value in dataset_stats.items():
    #             print(f"\t{key}: {value}")     


if __name__ == "__main__":
    # Script execution example:
    # python experiments/prepare_coco_subset.py 
    # --data_path "../datasets/coco2017/train2017" 
    # --annotations_path "../datasets/coco2017/annotations/instances_train2017.json"
    # --output_folder "../datasets/coco2017_cdt",
    # --categories cat dog train
    args = parse_args()

    generate_subset(data_path=args.data_path,
                    annotations_path=args.annotations_path,
                    output_path=args.output_path,
                    categories=args.categories)
