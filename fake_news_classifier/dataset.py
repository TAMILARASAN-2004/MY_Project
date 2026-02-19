from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class NewsExample:
    token_ids: List[int]
    image: torch.Tensor
    video: torch.Tensor
    label: int


class MultiModalNewsDataset(Dataset):
    def __init__(self, examples: List[NewsExample], max_text_len: int = 128):
        self.examples = examples
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        token_ids = ex.token_ids[: self.max_text_len]
        padded = token_ids + [0] * (self.max_text_len - len(token_ids))

        return {
            "token_ids": torch.tensor(padded, dtype=torch.long),
            "image": ex.image.float(),
            "video": ex.video.float(),
            "label": torch.tensor(ex.label, dtype=torch.long),
        }
