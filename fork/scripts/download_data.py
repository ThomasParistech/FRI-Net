"""Download preprocessed datasets for FRI-Net.

Checkpoints (pretrained_ckpt.pth, pretrained_scenecad_ckpt.pth,
pretrained_room_wise_encoder.pth) must be downloaded manually from Google Drive:
https://drive.google.com/drive/folders/1YMnVm2YkLin3Z6PmBkGMqEoSPx4WhdT

Usage:
    python fork/scripts/download_data.py
"""

import os
import subprocess
import zipfile

FORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":

    checkpoints_dir = os.path.join(FORK_ROOT, "input", "checkpoints")
    print("Datasets and Checkpoints must be downloaded manually from:")
    print("  stru3d: https://drive.google.com/file/d/1TgqNB59ZOqdTSJieNoeHR1XvwEfsuIzB ")
    print("  scenecad: https://drive.google.com/file/d/1rncl4e6KN0ZbkvyRjIuOgdaoIGsrwwQg ")
