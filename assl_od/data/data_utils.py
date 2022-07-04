from typing import Dict

from PIL import Image
from numpy import isin
from plotly import data

from torchvision.datasets import CocoDetection


def compute_statistics(dataset: CocoDetection) -> Dict:
    dataset_stats: Dict = None

    if isinstance(dataset, CocoDetection):
        dataset_stats = __compute_coco_statistics(dataset)
    else:
        raise ValueError("Dataset statistics can be extracted only for "
                         "COCODetection objects.")

    # Add dataset set to stats dictionary
    dataset_stats["images_count"] = len(dataset)

    return dataset_stats


# TODO: should all these things be part of a class?
def __compute_coco_statistics(dataset: CocoDetection) -> Dict:
    """Extracts information and statistics about the CocoDetection dataset
    provided as parameter.

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

    categories_count = len(dataset.coco.cats)
    dataset_stats["categories_count"] = categories_count

    categories = [(item["id"], item["name"]) 
                  for item in dataset.coco.cats.values()]
    dataset_stats["categories"] = categories

    supercategories = []
    supercategories = [item["supercategory"]
                       for item in dataset.coco.cats.values()]
    dataset_stats["supercategories_count"] = len(supercategories)                       
    dataset_stats["supercategories"] = list(set(supercategories))

    supercategory_mapping = {supercat: [] for supercat in supercategories}
    {supercategory_mapping[item["supercategory"]].append(item["name"])
     for item in dataset.coco.cats.values()}
    dataset_stats["supercategory_mapping"] = supercategory_mapping

    instances_count = sum([len(count)
                           for count in dataset.coco.catToImgs.values()])
    dataset_stats["instances_count"] = instances_count

    category_distrib = {
        dataset.coco.cats[cat_id]["name"]: 
        (len(img_ids), round(len(img_ids )/instances_count * 100, 2))
        for (cat_id, img_ids) in dataset.coco.catToImgs.items()}
    dataset_stats["category_distrib"] = category_distrib

    supercategory_distrib = {k: 0 for k in supercategories}
    for supercategory, categories in supercategory_mapping.items():
        for category in categories:
            supercategory_distrib[supercategory] += \
                category_distrib[category][0]
    supercategory_distrib = {k: (v, round(v/instances_count * 100, 2))
                             for (k, v) in supercategory_distrib.items()}
    dataset_stats["supercategory_distrib"] = supercategory_distrib

    return dataset_stats