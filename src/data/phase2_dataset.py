"""Tensor dataset for model-specific Phase 2 neural adapter outputs."""

import numpy as np
import torch
from torch.utils.data import Dataset


class Phase2TensorDataset(Dataset):
    def __init__(self, inputs, targets, row_ids):
        self.inputs = {}
        for name, values in inputs.items():
            array = np.asarray(values)
            dtype = torch.long if np.issubdtype(array.dtype, np.integer) else torch.float32
            self.inputs[name] = torch.as_tensor(array, dtype=dtype)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.row_ids = torch.as_tensor(row_ids, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        inputs = {name: values[index] for name, values in self.inputs.items()}
        inputs["row_ids"] = self.row_ids[index]
        return inputs, self.targets[index]
