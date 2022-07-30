# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from pathlib import Path

from detectron2.data.datasets import register_coco_instances


_SPLITS_COCO_CDT_FORMAT = {
    "coco_cdt_train": (
        "coco_cdt/annotations/instances_train2017.json",
        "coco_cdt/train2017/"
    ),
    "coco_cdt_test": (
        "coco_cdt/annotations/instances_val2017.json",
        "coco_cdt/val2017/"
    )
}


def register_coco_cdt_label(root):
    """
    Register the newly created 'coco_cdt' custom dataset, both training
    and test versions.
    """
    # Register train dataset
    register_coco_instances(
        name="coco_cdt_train",
        metadata={},
        json_file=Path(root) / _SPLITS_COCO_CDT_FORMAT["coco_cdt_train"][0],
        image_root=Path(root) / _SPLITS_COCO_CDT_FORMAT["coco_cdt_train"][1])

    # Register test dataset
    register_coco_instances(
        name="coco_cdt_test",
        metadata={},
        json_file=Path(root) / _SPLITS_COCO_CDT_FORMAT["coco_cdt_test"][0],
        image_root=Path(root) / _SPLITS_COCO_CDT_FORMAT["coco_cdt_test"][1])



_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_cdt_label(_root)
