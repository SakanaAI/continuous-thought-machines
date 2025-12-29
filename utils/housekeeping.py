import numpy as np
import random
import torch


import os
import zipfile
import glob

def zip_python_code(output_filename):
    """
    Zips all .py files in the current repository and saves it to the
    specified output filename.

    Args:
        output_filename: The name of the output zip file.
                         Defaults to "python_code_backup.zip".
    """

    with zipfile.ZipFile(output_filename, 'w') as zipf:
        files = glob.glob('models/**/*.py', recursive=True) + glob.glob('utils/**/*.py', recursive=True) + glob.glob('tasks/**/*.py', recursive=True) + glob.glob('*.py', recursive=True)
        for file in files:
            root = '/'.join(file.split('/')[:-1])
            nm = file.split('/')[-1]
            zipf.write(os.path.join(root, nm))

def set_seed(seed=42, deterministic=True):
    """
    ... and the answer is ...
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """Automatic convergence detection with patience-based stopping."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop based on validation score.

        Args:
            score: Current validation metric (higher is better)

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.early_stop = False
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    @property
    def should_stop(self) -> bool:
        """Return current stopping state."""
        return self.early_stop

    @property
    def best_score_so_far(self) -> float:
        """Return the best validation score observed."""
        return self.best_score if self.best_score is not None else float('-inf')
