import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

from assl_od.data.dataset_coco_processor import CocoDatasetProcessor


SEMISUP_PERCENTAGES = [0.5, 1., 2., 5., 10.]
RUNS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for the generation of the semi-supervised'
                    'data split for a COCO format dataset.')

    parser.add_argument("--data_path", type=Path, required=True,
                    help="Path to COCO images folder.")
    parser.add_argument("--annotations_path", type=Path, required=True,
                    help="Path to COCO annotations .json file.")
    parser.add_argument("--output_filepath", type=Path, required=True,
                        help="Path to output file where the split info "
                             "will be stored.")

    args = parser.parse_args()

    return args


def sample_data_uniform(semisupervised_percentage: float,
                        categories_to_imgs: Dict[int, List[int]],
                        dataset_size: int):
    """Given a dictionary containing the list of images in each
    category, uniformly sample images from each category. Repeats this
    process for a number of times equal to `RUNS` and stores sampled
    data info in a dictionary indexed by the run id."""

    data_dict = {}
    samples_count = int(semisupervised_percentage / 100. * dataset_size)

    for run_idx in range(RUNS):
        # For every run, sample uniformly from all categories
        all_samples = []
        for imgs in categories_to_imgs.values():
            labeled_count = int(semisupervised_percentage / 100. * len(imgs))
            samples = np.random.choice(np.array(imgs), size=labeled_count,
                                       replace=False)
            all_samples.extend(samples.tolist())
        
        all_samples = np.array(all_samples)
        np.random.shuffle(all_samples)
        data_dict[run_idx] = all_samples[:samples_count].tolist()

    return data_dict


def generate_semisupervised_split(data_path: Path, annotations_path: Path, 
                                  output_filepath: Path):
    """
    Loads the dataset specified by `data_path` and `annotations_path` and
    randomly generates subsets of labeled data to be used in the
    semi-supervised setup. Sampling is uniform across the categories of the
    dataset. Data subsets are generated for multiple labeled data percentages
    and stored, as dictionary, in a single output file.

    The structure of the resulting dictionary, stored at `output_filepath` is:
    {labeled_data_percentage_1: {0: list_1_of_imgs_with_cat_id_1,
                                 1: list_1_of_imgs_with_cat_id_2,
                                 ...},
     labeled_data_percentage_2: {0: list_2_of_imgs_with_cat_id_1,
                                 1: list_2_of_imgs_with_cat_id_2,
                                 ...},
     ...}

    Args:
        data_path: path to the folder where images are stored
        annotations_path: path to the annotations .json file
        output_filepath: path of the output file, where the selected labeled
            data for every percentage is stored
    """

    dset_processor = CocoDatasetProcessor(
        data_folder=data_path, annotations_file_path=annotations_path)

    # Get dictionary of lists that maps categories (by id) to images
    cat_to_imgs = dset_processor.dataset.coco.catToImgs

    # Randomly select the labeled data of the semi-supervised setup
    labeled_data = {}
    dataset_size = len(dset_processor.dataset.ids)
    for semisup_percentage in SEMISUP_PERCENTAGES:  
        labeled_data[semisup_percentage] = sample_data_uniform(
            semisupervised_percentage=semisup_percentage,
            categories_to_imgs=cat_to_imgs,
            dataset_size=dataset_size)

    # Store the resulting data split
    with open(output_filepath, "w") as file:
        json.dump(labeled_data, file)


if __name__ == "__main__":
    # Script execution example:
    # python experiments/split_coco_semisup.py 
    # --data_path "../datasets/coco/train2017" 
    # --annotations_path "../datasets/coco/annotations/instances_train2017.json"
    # --output_filepath "assl_od/unbiased-teacher-main/dataseed/COCO_cdt_supervision.txt"

    args = parse_args()

    generate_semisupervised_split(
        data_path=args.data_path,
        annotations_path=args.annotations_path,
        output_filepath=args.output_filepath,
    )



