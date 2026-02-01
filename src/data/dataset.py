import torch
from torch.utils.data import Dataset

class DAFDataset(Dataset):
    """
    PyTorch Dataset for DAF-MoE.
    Wraps preprocessed numerical, categorical idx, and categorical meta tensors.
    """
    def __init__(self, X_num, X_cat_idx, X_cat_meta, y=None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat_idx = torch.tensor(X_cat_idx, dtype=torch.long)
        self.X_cat_meta = torch.tensor(X_cat_meta, dtype=torch.float32)
        
        self.y = None
        if y is not None:
            # Classification/Regression both use float32 for task_criterion compatibility
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        inputs = {
            'x_numerical': self.X_num[idx],
            'x_categorical_idx': self.X_cat_idx[idx],
            'x_categorical_meta': self.X_cat_meta[idx]
        }
        if self.y is not None:
            return inputs, self.y[idx]
        return inputs