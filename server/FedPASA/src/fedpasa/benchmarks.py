from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PASCAL_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
)

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

NUS_CLASSES = (
    "airport", "animal", "beach", "bear", "birds", "boats", "book", "bridge",
    "buildings", "cars", "castle", "cat", "cityscape", "clouds", "computer",
    "coral", "cow", "dancing", "dog", "earthquake", "elk", "fire", "fish",
    "flags", "flowers", "food", "fox", "frost", "garden", "glacier", "grass",
    "harbor", "horses", "house", "lake", "leaf", "map", "military", "moon",
    "mountain", "nighttime", "ocean", "person", "plane", "plants", "police",
    "protest", "railroad", "rainbow", "reflection", "road", "rocks", "running",
    "sand", "sign", "sky", "snow", "soccer", "sports", "statue", "street", "sun",
    "sunset", "surf", "swimmers", "tattoo", "temple", "tiger", "tower", "town",
    "toy", "train", "tree", "valley", "vehicle", "water", "waterfall", "wedding",
    "whales", "window", "zebra",
)


@dataclass(frozen=True)
class Benchmark:
    name: str
    classes: tuple[str, ...]
    train_list: str
    test_list: str
    expected_train: int
    expected_test: int
    purification_temperature: float
    cls_scale: float
    patch_scale: float
    piaa_reported_map: float
    path_mode: str

    @property
    def class_count(self) -> int:
        return len(self.classes)

    def image_name(self, identifier: str) -> str:
        value = Path(identifier).name if self.path_mode == "flat" else identifier
        return value if value.lower().endswith(".jpg") else f"{value}.jpg"

    def client_counts(self) -> tuple[int, ...]:
        return tuple(max(2, int(round(self.class_count * ratio))) for ratio in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0))


BENCHMARKS = {
    "voc07": Benchmark(
        "voc07", PASCAL_CLASSES, "voc07train_labels.txt", "voc07val_labels.txt",
        5011, 4952, 100.0, 80.0, 200.0, 92.5, "direct",
    ),
    "voc12": Benchmark(
        "voc12", PASCAL_CLASSES, "voc12train_labels.txt", "voc12val_labels.txt",
        5717, 5823, 100.0, 80.0, 200.0, 92.2, "direct",
    ),
    "coco": Benchmark(
        "coco", COCO_CLASSES, "cocotrain_labels.txt", "cocoval_labels.txt",
        82081, 40137, 0.01, 80.0, 400.0, 73.2, "direct",
    ),
    "nus": Benchmark(
        "nus", NUS_CLASSES, "nuswidetrain_labels.txt", "nuswideval_labels.txt",
        150000, 60260, 500.0, 10.0, 400.0, 50.6, "flat",
    ),
}


FEDMPT_MAP = {
    "voc07": (90.13, 90.12, 90.01, 90.00, 89.61, 87.19),
    "coco": (67.37, 65.91, 65.31, 63.86, 63.09, 62.37),
    "nus": (58.36, 57.27, 57.03, 56.52, 56.08, 54.87),
}


# mAP columns from FedMPT Table 1. These are external published results, not
# re-runs in this repository; storing the full matrix makes the comparison and
# strongest-baseline audit reproducible.
EXTERNAL_MAPS = {
    "voc07": {
        "Fed-DualCoOp (NeurIPS 2022)": (87.67, 85.98, 85.73, 84.46, 83.76, 82.02),
        "Fed-SCPNet (CVPR 2023)": (83.37, 81.15, 80.03, 79.89, 76.51, 76.48),
        "Fed-MaPLe (CVPR 2023)": (84.22, 87.87, 84.36, 81.82, 80.71, 76.08),
        "FedPGP (ICML 2024)": (85.47, 84.51, 84.47, 79.89, 76.60, 78.16),
        "Fed-TCP (CVPR 2024)": (81.56, 81.15, 80.65, 77.32, 76.50, 76.42),
        "FedTPG (ICLR 2024)": (88.11, 85.23, 85.95, 84.23, 84.45, 82.72),
        "Fed-PosCoOp (WACV 2025)": (87.42, 84.71, 82.53, 81.65, 80.71, 79.90),
        "FedAWA (CVPR 2025)": (84.75, 83.17, 81.77, 79.48, 78.81, 78.22),
        "Fed-RAM (CVPR 2025)": (88.05, 86.24, 85.50, 85.68, 85.53, 83.04),
        "FedMVP (ICCV 2025)": (87.14, 86.86, 86.64, 85.61, 84.87, 82.43),
        "FedMPT (CVPR 2026)": FEDMPT_MAP["voc07"],
    },
    "coco": {
        "Fed-DualCoOp (NeurIPS 2022)": (61.88, 58.39, 55.95, 53.80, 52.28, 51.46),
        "Fed-SCPNet (CVPR 2023)": (60.96, 57.68, 52.79, 50.10, 48.40, 46.67),
        "Fed-MaPLe (CVPR 2023)": (64.83, 62.98, 61.80, 60.81, 55.83, 47.24),
        "FedPGP (ICML 2024)": (63.10, 59.86, 58.21, 58.18, 56.52, 54.53),
        "Fed-TCP (CVPR 2024)": (62.66, 62.79, 61.77, 60.07, 59.99, 57.22),
        "FedTPG (ICLR 2024)": (63.91, 61.34, 60.95, 60.54, 54.53, 50.88),
        "Fed-PosCoOp (WACV 2025)": (63.28, 59.94, 55.88, 53.23, 50.95, 50.02),
        "FedAWA (CVPR 2025)": (65.58, 62.09, 61.47, 60.14, 59.01, 57.57),
        "Fed-RAM (CVPR 2025)": (66.06, 63.08, 59.97, 60.85, 60.88, 55.64),
        "FedMVP (ICCV 2025)": (65.76, 63.19, 62.26, 60.60, 59.56, 58.48),
        "FedMPT (CVPR 2026)": FEDMPT_MAP["coco"],
    },
    "nus": {
        "Fed-DualCoOp (NeurIPS 2022)": (51.02, 49.97, 46.25, 42.69, 40.65, 39.40),
        "Fed-SCPNet (CVPR 2023)": (49.56, 47.61, 43.31, 39.28, 38.61, 38.49),
        "Fed-MaPLe (CVPR 2023)": (54.78, 53.68, 51.34, 51.62, 50.98, 51.20),
        "FedPGP (ICML 2024)": (55.92, 53.41, 53.78, 49.64, 48.45, 43.20),
        "Fed-TCP (CVPR 2024)": (54.46, 51.93, 50.91, 48.28, 47.20, 48.89),
        "FedTPG (ICLR 2024)": (53.01, 52.78, 50.00, 50.40, 49.92, 49.46),
        "Fed-PosCoOp (WACV 2025)": (52.88, 50.64, 48.77, 42.39, 40.62, 40.08),
        "FedAWA (CVPR 2025)": (54.67, 53.91, 50.40, 52.87, 52.75, 50.71),
        "Fed-RAM (CVPR 2025)": (56.35, 54.05, 53.51, 52.25, 52.61, 51.20),
        "FedMVP (ICCV 2025)": (53.20, 53.32, 51.90, 52.98, 51.33, 51.08),
        "FedMPT (CVPR 2026)": FEDMPT_MAP["nus"],
    },
}
