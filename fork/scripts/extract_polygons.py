#!/usr/bin/python3
"""Run FRI-Net evaluation and extract predicted polygons as JSON files.

FRI-Net uses separate eval scripts per dataset (eval_stru3d / eval_scenecad).
Each script's evaluate() saves per-scene .npy files containing room polygons.
We hook into the pipeline via `scene_polys_out` the same way as the other models.

Usage:
    python3 fork/scripts/extract_polygons.py
"""

import json
import os

from eval_stru3d import get_args_parser as get_args_parser_stru3d
from eval_stru3d import main as eval_main_stru3d
from eval_scenecad import get_args_parser as get_args_parser_scenecad
from eval_scenecad import main as eval_main_scenecad

FORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    "stru3d": dict(
        get_args_parser=get_args_parser_stru3d,
        eval_main=eval_main_stru3d,
        img_folder=os.path.join(FORK_ROOT, "input/stru3d/input"),
        occ_folder=os.path.join(FORK_ROOT, "input/stru3d/occ"),
        ids_path=os.path.join(FORK_ROOT, "input/stru3d"),
        checkpoint=os.path.join(FORK_ROOT, "input/checkpoints/pretrained_ckpt.pth"),
        canonical_name="STRUCTURED_3D",
    ),
    "scenecad": dict(
        get_args_parser=get_args_parser_scenecad,
        eval_main=eval_main_scenecad,
        img_folder=os.path.join(FORK_ROOT, "input/scenecad/input"),
        occ_folder=os.path.join(FORK_ROOT, "input/scenecad/occ"),
        ids_path=os.path.join(FORK_ROOT, "input/scenecad"),
        checkpoint=os.path.join(FORK_ROOT, "input/checkpoints/pretrained_scenecad_ckpt.pth"),
        canonical_name="SCENE_CAD",
    ),
}


def main(dataset_name: str) -> None:
    output_dir = os.path.join(FORK_ROOT, "output", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    dataset_cfg = DATASET_CONFIGS[dataset_name]

    parser = dataset_cfg["get_args_parser"]()
    args = parser.parse_args([])
    args.dataset_name = dataset_name
    args.img_folder = dataset_cfg["img_folder"]
    args.occ_folder = dataset_cfg["occ_folder"]
    args.ids_path = dataset_cfg["ids_path"]
    args.checkpoint = dataset_cfg["checkpoint"]

    # evaluate will populate this dict with {scene_id: [[x,y],...]}
    scene_polys: dict[str, list] = {}
    args.scene_polys_out = scene_polys

    dataset_cfg["eval_main"](args)

    out_path = os.path.join(output_dir, "polygons.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset_cfg["canonical_name"], "model": "FRI_Net", "data": scene_polys}, f)
    print(f"Saved {len(scene_polys)} scene predictions to {out_path}")


if __name__ == "__main__":
    for dataset_name in DATASET_CONFIGS:
        main(dataset_name)
