import math
import numpy as np
import logging
import os
import json
import time
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
import shelve

logger = logging.getLogger(__name__)


class LineByLineBotDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int, local_rank=-1, conv_sampling=True):
        logger.info(f"Creating features from dataset file at {file_path}")
        self.data = np.array(json.load(open(file_path, "rt")))
        self.data = [x for x in self.data if len(x) > 5]
        if conv_sampling:
            self.conv_weights = np.ones((len(self.data),)) / len(self.data)
        else:
            self.conv_weights = np.array([len(x) for x in self.data]) / np.sum([len(x) for x in self.data])
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.dblen = np.sum([len(x) for x in self.data])

    def __len__(self):
        return self.dblen

    def __getitem__(self, i) -> torch.Tensor:
        conv = np.random.choice(self.data, p=self.conv_weights)
        start = np.random.randint(len(conv))
        end = start + np.random.randint(2, 6)
        conv = conv[start:end]

        line = ""
        for x in conv:
            if x['owner']:
                line += f"</s> {x['content']}"
            else:
                line += f"<s> {x['content']}"
        return torch.tensor(self.tokenizer.encode(line, add_special_tokens=False, max_length=self.block_size),
                            dtype=torch.long)
