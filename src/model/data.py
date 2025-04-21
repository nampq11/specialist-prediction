import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MedicalDataFrameDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = 128
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        reason_text = str(row['reason_combind'])  # Ensure reason_text is a string
        user_info = torch.tensor([row['gender'], row['age_category']], dtype=torch.float32)
        label = row['specialist_name']
        encoded = self.tokenizer(
            [reason_text],  # Wrap reason_text in a list to ensure correct input type
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            "reason_text_ids": encoded['input_ids'].squeeze(0),
            "reason_text_mask": encoded['attention_mask'].squeeze(0),
            "user_info": user_info,
            "labels": torch.tensor(label, dtype=torch.long)
        }
