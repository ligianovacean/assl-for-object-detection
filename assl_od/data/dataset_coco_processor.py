import json
from pathlib import Path
from typing import Dict, List

from assl_od.data.dataset_processor import DatasetProcessor

import torchvision.datasets as dset


class CocoDatasetProcessor(DatasetProcessor):
    def __init__(self, data_folder: Path, annotations_file_path: Path):
        super().__init__(data_folder, annotations_file_path)

    def get_category_ids(self, category_names: List[str]) -> int:
        """Given a list of category names, returns the corresponding list of
        unique ids."""

        return [item["id"]
                for item in self.dataset.coco.cats.values()
                if item["name"] in category_names]

    def extract_annotations(self, categories: List[int]) -> Dict:
        """
        Given a list of category ids of interest, the annotations dictionary
        is generated such that it contains only instances of the specified
        categories.
        """

        # Read the annotations file of the source dataset
        with open(self.annotations_file_path, "r") as f:
            all_annotations = json.load(f)  

        # Initialize annotations dict and store generic information
        annotations = {}
        annotations["info"] = all_annotations["info"]
        annotations["licenses"] = all_annotations["licenses"] 
        
        # Set categories of the new subset
        categories_list = []
        for category_info in all_annotations["categories"]:
            if category_info["id"] in categories:
                categories_list.append(category_info)
        annotations["categories"] = categories_list

        # Select instances of the categories of interest and store
        # a list with the corresponding image ids
        annotations_list = []
        image_ids = []
        for annotation_info in all_annotations["annotations"]:
            if annotation_info['category_id'] in categories:
                annotations_list.append(annotation_info)
                image_ids.append(annotation_info["image_id"])
        annotations["annotations"] = annotations_list

        # Set the list of images in the dataset
        images_list = []
        image_ids = list(set(image_ids))
        for image_info in all_annotations["images"]:
            if image_info["id"] in image_ids:
                images_list.append(image_info)
        annotations["images"] = images_list

        return annotations

    def compute_statistics(self) -> Dict:
        """Extracts information and statistics about the CocoDetection
        dataset provided as parameter.

        The following informations is extracted:
        - number of categories
        - id to category name mapping
        - list of supercategories

        Args:
            dataset: the CocoDetection dataset object
        
        Returns: 
            Dictionary containing the statistics and info about the dataset
        """

        dataset_stats: Dict = {}

        # Count images/samples in the dataset
        dataset_stats["images_count"] = len(self.dataset)

        # Collect all classes/categories as id-name pairs
        categories_count = len(self.dataset.coco.cats)
        dataset_stats["categories_count"] = categories_count

        categories = [(item["id"], item["name"]) 
                    for item in self.dataset.coco.cats.values()]
        dataset_stats["categories"] = categories

        # Collect the super-categories list 
        supercategories = list(set(
            [item["supercategory"]
            for item in self.dataset.coco.cats.values()]))                       
        dataset_stats["supercategories"] = supercategories
        dataset_stats["supercategories_count"] = len(supercategories)

        # Collect, for all super-categories, the belonging categories
        supercategory_mapping = {supercat: [] for supercat in supercategories}
        {supercategory_mapping[item["supercategory"]].append(item["name"])
        for item in self.dataset.coco.cats.values()}
        dataset_stats["supercategory_mapping"] = supercategory_mapping

        # Count the total number of instances in the dataset images
        instances_count = sum([len(count)
                            for count in self.dataset.coco.catToImgs.values()])
        dataset_stats["instances_count"] = instances_count

        # Compute the pdf of instances across all categories
        category_distrib = {
            self.dataset.coco.cats[cat_id]["name"]: 
            (len(img_ids), round(len(img_ids )/instances_count * 100, 2))
            for (cat_id, img_ids) in self.dataset.coco.catToImgs.items()}
        dataset_stats["category_distrib"] = category_distrib

        # Compute the pdf of instances across all super-categories
        supercategory_distrib = {k: 0 for k in supercategories}
        for supercategory, categories in supercategory_mapping.items():
            for category in categories:
                supercategory_distrib[supercategory] += \
                    category_distrib[category][0]
        supercategory_distrib = {k: (v, round(v/instances_count * 100, 2))
                                for (k, v) in supercategory_distrib.items()}
        dataset_stats["supercategory_distrib"] = supercategory_distrib

        return dataset_stats


    def _read_dataset(self) -> dset.VisionDataset:
        """
        Reads a COCO dataset as follows:
        - images are read from the `data_path` folder
        - corresponding annotations are read from the `annotations_path` json

        Returns:
            The loaded COCO dataset, stored as a torchvision.datasets.CocoDetection
            object.
        """

        data = dset.CocoDetection(root=self.data_folder, annFile=self.annotations_file_path)

        return data