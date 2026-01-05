import torch
import torch.nn as nn
import torch.nn.functional as F

class CodeCNN(nn.Module):
    def __init__(self, input_dim, filters=32, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, filters, k) for k in kernel_sizes
        ])
        self.output_dim = filters * len(kernel_sizes)
        
    def forward(self, x):
        # x: [batch, seq_len, feat_dim] -> [batch, feat_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        outputs = []
        for conv in self.convs:
            # Conv1d -> ReLU -> MaxPool
            out = F.relu(conv(x))
            out = F.max_pool1d(out, out.shape[2]) # Global Max Pooling
            outputs.append(out.squeeze(2))
            
        return torch.cat(outputs, dim=1) # [batch, filters*3]

class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.output_dim = hidden_dim // 2
        
    def forward(self, x):
        return self.net(x)

class HybridTrustModel(nn.Module):
    def __init__(self, token_feat_dim, mlp_input_dim, cnn_filters=32, mlp_hidden=64):
        super().__init__()
        
        self.cnn_branch = CodeCNN(token_feat_dim, filters=cnn_filters)
        self.mlp_branch = FeatureMLP(mlp_input_dim, hidden_dim=mlp_hidden)
        
        fusion_dim = self.cnn_branch.output_dim + self.mlp_branch.output_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.code_head = nn.Linear(32, 1)
        self.summary_head = nn.Linear(32, 1)
        
    def forward(self, token_feats, mlp_feats):
        cnn_out = self.cnn_branch(token_feats)
        mlp_out = self.mlp_branch(mlp_feats)
        
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        fused = self.fusion_layer(combined)
        
        # Logits
        p_code_logit = self.code_head(fused)
        p_summary_logit = self.summary_head(fused)
        
        return p_code_logit, p_summary_logit

def calculate_trust_score(p_code_prob, p_summary_prob, s_api, w1=0.4, w2=0.4, w3=0.2):
    """
    Computes Trust Score = w1(1 - P_code) + w2(1 - P_summary) + w3(S_api)
    """
    trust = w1 * (1 - p_code_prob) + w2 * (1 - p_summary_prob) + w3 * s_api
    return trust
