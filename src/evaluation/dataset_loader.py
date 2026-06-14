"""
Loads and validates dataset.json into typed QAPair models.
"""

import json
from pathlib import Path
from src.evaluation.models import QAPair


DATASET_PATH = Path(__file__).parent / "dataset.json"


def load_dataset(path: Path = DATASET_PATH) -> list[QAPair]:
    """
    Load dataset.json and return a validated list of QAPair objects.

    Raises:
        FileNotFoundError: if dataset.json doesn't exist at the given path.
        ValueError: if any entry fails Pydantic validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))

    dataset = [QAPair(**entry) for entry in raw]

    print(f"[dataset_loader] Loaded {len(dataset)} QA pairs from {path.name}")
    return dataset