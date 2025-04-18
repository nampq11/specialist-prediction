import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score

class BertClassifier(nn.Module):
    def __init__(self, model_name, dropout, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for param in self.bert.parameters():
            param.requires_grad = False # Freeze BERT parameters
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc_classifier = nn.Sequential(
            nn.Linear(768, 512),
            self.relu,
            nn.Linear(512, num_classes)
        )

    # def forward(self, input_ids, attention_mask):
    #     bert_output, _ = self.bert(input_ids=input_ids, 
    #                                attention_mask=attention_mask,
    #                                return_dict=False)
    #     fc_output = self.fc_classifier(bert_output)
    #     return fc_output

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, 
                               attention_mask=attention_mask)
        
        dimension = 768
        outputs = bert_output.last_hidden_state[:, 0][:dimension]
        outputs = F.normalize(outputs, p=2, dim=1)
        fc_output = self.fc_classifier(outputs)
        return fc_output

    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs