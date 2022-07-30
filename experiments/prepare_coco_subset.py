import os
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torchvision.datasets as dset

from assl_od.data.dataset_coco_processor import CocoDatasetProcessor


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
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="If specified, dataset statistics are displayed.")                   

    args = parser.parse_args()

    return args


def prepare_subset_folders(data_path: Path, annotations_path: Path,
                           output_folder_root: Path):
    """Generates the folders where the new dataset will be stored."""

    # Create images folder if is does not exist
    imgs_folder = output_folder_root / data_path.name
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)

    # Create annotations folder if it does not exist
    annotations_folder = output_folder_root / annotations_path.parent.name
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)

    return imgs_folder, annotations_folder


def display_dataset_stats(dataset_stats):
    """Prints dataset stats."""

    print(f"\n\nDataset statistics:\n")
    for key, value in dataset_stats.items():
        print(f"\t{key}: {value}\n")  


def copy_images(src_folder: Path, dst_folder: Path, image_names: List[str]):
    """
    Copies the images in the `image_names` list from `src_folder` to
    `dst_folder`.
    """

    for img_name in image_names:
        shutil.copyfile(src_folder / img_name, dst_folder / img_name)


def generate_subset(data_path: Path, annotations_path: Path, 
                    output_path: Path, categories: List[str],
                    verbose: bool = False):
    """
    Extracts from the dataset at `data_path` the samples belonging to
    the specified categories, along with the corresponding annotations.
    The resulting subset is stored at `output_path`, following the same
    folder struscture as the original dataset.
    """     

    dset_processor = CocoDatasetProcessor(
        data_folder=data_path, annotations_file_path=annotations_path)

    # Read original COCO dataset
    if verbose:
        dataset_stats = dset_processor.compute_statistics()
        display_dataset_stats(dataset_stats)

    # Prepare output folders
    imgs_folder, annotations_folder = prepare_subset_folders(
        data_path=data_path, annotations_path=annotations_path,
        output_folder_root=output_path)
   
    # Get the category ids of the categories of interest, based on name
    category_ids = dset_processor.get_category_ids(category_names=categories)

    # Extract the updated annotations json that contains only
    # instances of specified categories
    print("Extracting data subset annotations...")
    subset_annotations_dict = dset_processor.extract_annotations(
        categories=category_ids)

    # Store updated annotations as json 
    print("Storing annotations...")
    annotations_out_filepath = annotations_folder / annotations_path.name
    with open(annotations_out_filepath, "w") as out_file:
        json.dump(subset_annotations_dict, out_file)

    # Store the images of the new data subset
    print("Storing images...")
    image_names = [item["file_name"] 
                   for item in subset_annotations_dict["images"]]
    copy_images(
        src_folder=data_path, dst_folder=imgs_folder,
        image_names=image_names)

    print(f"Done data subset generation for categories {categories}.")


if __name__ == "__main__":
    # Script execution example:
    # python experiments/prepare_coco_subset.py 
    # --data_path "../datasets/coco/train2017" 
    # --annotations_path "../datasets/coco/annotations/instances_train2017.json"
    # --output_path "../datasets/coco_cdt/coco/",
    # --categories cat dog train
    
    args = parse_args()

    generate_subset(data_path=args.data_path,
                    annotations_path=args.annotations_path,
                    output_path=args.output_path,
                    categories=args.categories,
                    verbose=args.verbose)
