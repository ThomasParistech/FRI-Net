#!/usr/bin/python3
"""Download data for FRI-Net."""

import os
import tempfile
import zipfile

from helper_3dml.io.download import download_gdrive

from helper_3dml.io.file import move_folder
from helper_3dml.io.file import move_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    assert os.path.basename(ROOT) == "FRI-Net"
    # Montefloor Eval
    download_gdrive("1jQ8WMwkk4FmgRdMrAPQc9MIiwPt3K7-s", os.path.join(ROOT, "s3d_floorplan_eval"), unzip=True)

    # Preprocessed data + Checkpoints (Structured3D)
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_gdrive("1TgqNB59ZOqdTSJieNoeHR1XvwEfsuIzB", tmp_dir, unzip=True)
        for name in ["pretrained_ckpt.pth", "pretrained_room_wise_encoder.pth"]:
            move_file(os.path.join(tmp_dir, name), os.path.join(ROOT, "checkpoints", name))

        with zipfile.ZipFile(os.path.join(tmp_dir, "stru3d.zip")) as zf:
            zf.extractall(os.path.join(ROOT, "data", "stru3d"))

    # Preprocessed data + Checkpoints (SceneCAD)
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_gdrive("1rncl4e6KN0ZbkvyRjIuOgdaoIGsrwwQg", tmp_dir, unzip=True)
        name = "pretrained_scenecad_ckpt.pth"
        move_file(os.path.join(tmp_dir, name), os.path.join(ROOT, "checkpoints", name))

        move_folder(os.path.join(tmp_dir, "data"), os.path.join(ROOT, "data", "scenecad"))
