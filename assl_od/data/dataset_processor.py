from typing import Dict, List
from pathlib import Path
from abc import ABC, abstractmethod

from PIL import Image
from numpy import isin
from plotly import data

import torchvision.datasets as dset


class DatasetProcessor(ABC):
    def __init__(self, data_folder: Path, annotations_file_path: Path):
        self.data_folder = data_folder
        self.annotations_file_path = annotations_file_path

        self.dataset = self._read_dataset()

    @abstractmethod
    def get_category_ids(self, category_names: List[str]) -> int:
        pass

    @abstractmethod
    def extract_annotations(self, categories: List[int]) -> Dict:
        pass

    @abstractmethod
    def compute_statistics(self) -> Dict:
        pass

    @abstractmethod
    def _read_dataset(self) -> dset.VisionDataset:
        pass