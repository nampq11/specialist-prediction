import torch
from typing import List, Dict, Any

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokens: List, labels: List):
        """
        Initialize a TextDataset.

        Args:
            tokens (list): List of tokenized text data.
            labels (list): List of corresponding labels.
        This class is desiged to work as PyTorch Dataset, which means it can used with Pytoch's DataLoader for efficient data loading during training and evaluation.
        """
        self.tokens = tokens
        self.labels = labels
    
    def __getitem__(self, idx):
        """
        Get a specific data point (a pair of text data and its label) from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.
        
        Returns:
            tuple: A tuple containing the label and tokenized text data for the specified data point.
        """
        return self.labels[idx], self.tokens[idx]

    def __len__(self):
        """
        Get the total number of data points in the dataset.
        Returns:
            int: Number of data points in the dataset.
        """
        return len(self.tokens)