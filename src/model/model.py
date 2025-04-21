import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch.nn.functional as F



class MedicalSpecialistClassifer(nn.Module):
    def __init__(self, num_specialists: int, model_name: str = "BookingCare/gte-multilingual-base-v2.1", user_feature_dim=2, dropout=0.2, load_pretrained=True, trust_remote_code=False):
        super(MedicalSpecialistClassifer, self).__init__()

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if load_pretrained:
            self.reason_encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        else:
            self.reason_encoder = AutoModel.from_config(config, trust_remote_code=True)
        
        for param in self.reason_encoder.parameters():
            param.requires_grad = False
        self.reason_encoder_hidden_dim = self.reason_encoder.config.hidden_size
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, 128),
            nn.BatchNorm1d(128),
            self.relu,
            self.dropout,
            nn.Linear(128, 256),
            self.relu
        )

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(self.reason_encoder_hidden_dim, 512),
            nn.BatchNorm1d(512),
            self.relu,
            self.dropout,
        )

        self.level1_output = nn.Linear(512, num_specialists)

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(self.reason_encoder_hidden_dim + 256 + 512, 768),
            nn.BatchNorm1d(768),
            self.relu,
            self.dropout
        )
        self.level2_output = nn.Linear(768, num_specialists)

    def forward(self, reason_text_ids, reason_text_mask, user_info=None):
        reason_outputs = self.reason_encoder(reason_text_ids, attention_mask=reason_text_mask)
        reason_embedding = reason_outputs.last_hidden_state[:, 0, :]

        hidden1 = self.hidden_layer1(reason_embedding)

        level1_logits = self.level1_output(hidden1)

        if user_info is None:
            return level1_logits, None

        user_features = self.user_encoder(user_info)

        combined_features = torch.cat((reason_embedding, hidden1, user_features), dim=1)

        hidden2 = self.hidden_layer2(combined_features)
        level2_logits = self.level2_output(hidden2)

        return level1_logits, level2_logits

    def predict(self, reason_text_ids, reason_text_mask, user_info=None, threshold=0.7):
        self.eval()
        with torch.no_grad():
            level1_logits, level2_logits = self.forward(reason_text_ids, reason_text_mask, user_info)
            level1_probs = F.softmax(level1_logits, dim=1)
            max_probs, preds = torch.max(level1_probs, dim=1)
            final_preds = preds.clone()

            if user_info is not None and level2_logits is not None:
                for i, prob in enumerate(max_probs):
                    max_prob, pred = torch.max(level2_logits[i], dim=0)
                    final_preds[i] = pred
                    max_probs[i] = max_prob
                
                return final_preds, max_probs
            
            return final_preds, max_probs