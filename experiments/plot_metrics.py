import argparse
import json
import os
from pathlib import Path
from turtle import color
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from strenum import StrEnum


class Stage(StrEnum):
    TRAIN = "train"
    EVALUATION = "val"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for generating metrics plot based on a '
                    '.json file where metrics are logged during training.')

    parser.add_argument("--metrics_json", type=Path, required=True,
                        help="Path to COCO images folder.") 
    parser.add_argument("--folder", type=str, default="plots",
                        help="Name of the folder where plots are stored.")
    parser.add_argument("--train_step", type=int, default=20,
                        help="Specifies how often training metrics are "
                             "logged.")
    parser.add_argument("--eval_step", type=int, default=100,
                        help="Specifies how often evaluation metrics are "
                             "logged.")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                        help="Categories for which AP is computed.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Plot metrics starting with specified index.")

    args = parser.parse_args()

    return args


def get_stage(metrics: Union[List[str], str]) -> str:
    if "bbox/AP" in metrics:
        return Stage.EVALUATION

    return Stage.TRAIN


def parse_json(filepath: Path, start_index: int) -> Dict[str, List]:
    """
    Parses the .json file at `filepath` and returns a dictionary where:
    * keys are metric names
    * values are lists with all the values for that metric
    """
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]

    data = data[start_index:]

    metrics: Dict[str, List] = {}
    for step_metrics in data:
        for metric_name, value in step_metrics.items():
            # Get stage (train or eval)
            stage = get_stage(metric_name)
            # Prefix metric with stage information
            metric_name = f"{stage}_{metric_name}"

            if metric_name in metrics.keys():
                metrics[metric_name].append(value)
            else:
                metrics[metric_name] = [value]

    return metrics


def plot_stage_wise_metrics(names: List[str], metrics: Dict[str, List],
         folder: Path, out_filename: str,  
         stages_info: Dict[str, Tuple[int, int, str]], start_index: int):
    _, ax = plt.subplots(1, len(names), figsize=(len(names) * 10, 10))

    for stage, (step_size, start_index, color) in stages_info.items():
        for idy, metric in enumerate(names):
            name = f"{stage}_{metric}"
            y_data = metrics[name]
            x_data = np.arange(step_size*start_index, 
                               step_size*(len(y_data)+start_index),
                               step_size)

            c_axis = ax[idy] if len(names) > 1 else ax
            c_axis.plot(x_data, y_data, label=name, color=color)
            c_axis.set_xlabel('Iterations')
            c_axis.grid(axis='y', color='black', alpha=0.3)
            c_axis.grid(axis='x', color='black', alpha=0.3)
            c_axis.legend()

    plt.savefig(folder / out_filename, bbox_inches='tight')


def plot_metrics(metrics_json: Path, out_folder: str, train_step: int,
                 val_step: int, categories: List[str], start_index: int):
    """
    Parses the input .json file with during-training information and generates
    the following plots for the following metrics:
    * total_loss
    * loss_cls
    * loss_box_reg
    * loss_rpn_cls
    * loss_rpn_loc
    * lr
    * bbox/AP
    * bbox/AP50
    * bbox/AP75
    * bbox/AP1
    * bbox/APm
    * bbox/APs
    * bbox/AP-<category> for all categories

    Plots are stored in the same folder as the input .json file, named as the
    metric they represent.
    """

    # Parse json and store metrics in dictionary
    metrics = parse_json(metrics_json, start_index)

    # Create output folder if it does not exist
    output_folder = metrics_json.parent / out_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_start_index = start_index + 1
    val_start_index = start_index / (val_step / train_step) + 1
    train_stage_info = {Stage.TRAIN: (train_step, train_start_index, 'b')}
    val_stage_info = {Stage.EVALUATION: (val_step, val_start_index, 'r')}

    # Plot train and validation losses
    plot_stage_wise_metrics(
        names=["total_loss", "loss_box_reg", "loss_cls", "loss_rpn_cls",
               "loss_rpn_loc"],
        metrics=metrics,
        folder=output_folder,
        out_filename=f"losses:{start_index}.png",
        stages_info=train_stage_info,
        start_index=start_index
    )

    # Plot training learning rate
    plot_stage_wise_metrics(
        names=["lr"],
        metrics=metrics,
        folder=output_folder,
        out_filename=f"lr:{start_index}.png",
        stages_info=train_stage_info,
        start_index=start_index
    )

    # Plot cross-category validation AP metrics
    plot_stage_wise_metrics(
        names=["bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/APl", "bbox/APm",
               "bbox/APs"],
        metrics=metrics,
        folder=output_folder,
        out_filename=f"AP:{start_index}.png",
        stages_info=val_stage_info,
        start_index=start_index
    )

    # Plot category-wise validation AP metrics
    plot_stage_wise_metrics(
        names=[f"bbox/AP-{category}" for category in categories],
        metrics=metrics,
        folder=output_folder,
        out_filename=f"AP_class-wise:{start_index}.png",
        stages_info=val_stage_info,
        start_index=start_index
    )


if __name__ == "__main__":
    # Script execution example:
    # python experiments/plot_metrics.py 
    # --metrics_json "./output/baseline_coco_cdt/metrics.json"
    # [--folder "plots"]
    # [--train_step 20] 
    # [--eval_step 100]
    # [--categories cat dog train]
    # [--start_index 10]
    
    args = parse_args()

    plot_metrics(
        metrics_json=args.metrics_json,
        out_folder=args.folder,
        train_step=args.train_step,
        val_step=args.eval_step,
        categories=args.categories,
        start_index=args.start_index)