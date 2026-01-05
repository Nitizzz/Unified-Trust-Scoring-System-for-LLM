import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import spearmanr

from data_loading import TrustDataset
from model import HybridTrustModel, calculate_trust_score
import features

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    full_dataset = TrustDataset()
    
    # --- Group Split ---
    # We need to split by 'id' to prevent data leakage (variants of same question)
    # Extract IDs from the internal dataframe
    if hasattr(full_dataset, 'df'):
        all_ids = full_dataset.df['id'].unique()
    else:
        # Fallback if dataset implementation changes
        print("Warning: Could not access dataframe for group splitting. Using random split.")
        all_ids = np.arange(len(full_dataset)) 

    np.random.seed(cfg['training']['seed'])
    np.random.shuffle(all_ids)
    
    split_idx = int(len(all_ids) * 0.8)
    train_ids = set(all_ids[:split_idx])
    test_ids = set(all_ids[split_idx:])
    
    train_indices = [i for i, row in full_dataset.df.iterrows() if row['id'] in train_ids]
    test_indices = [i for i, row in full_dataset.df.iterrows() if row['id'] in test_ids]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"Train samples: {len(train_dataset)} (Groups: {len(train_ids)})")
    print(f"Test samples: {len(test_dataset)} (Groups: {len(test_ids)})")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    # --- Weight Calculation ---
    # Calculate positive weights for loss function
    train_labels_code = full_dataset.df.iloc[train_indices]['label_code'].values
    train_labels_summ = full_dataset.df.iloc[train_indices]['label_summary'].values
    
    num_pos_c = np.sum(train_labels_code)
    num_neg_c = len(train_labels_code) - num_pos_c
    pos_weight_c = torch.tensor(num_neg_c / max(num_pos_c, 1), dtype=torch.float32).to(device)
    
    num_pos_s = np.sum(train_labels_summ)
    num_neg_s = len(train_labels_summ) - num_pos_s
    pos_weight_s = torch.tensor(num_neg_s / max(num_pos_s, 1), dtype=torch.float32).to(device)
    
    print(f"Code Positives: {num_pos_c}, Negatives: {num_neg_c}, Weight: {pos_weight_c.item():.2f}")
    
    # Model Setup
    sample = full_dataset[0]
    token_feat_dim = sample['token_features'].shape[1]
    mlp_feat_dim = sample['mlp_features'].shape[0]
    
    model = HybridTrustModel(token_feat_dim, mlp_feat_dim, 
                             cnn_filters=cfg['model']['cnn_filters'],
                             mlp_hidden=cfg['model']['mlp_hidden']).to(device)
    
    check_opt = torch.compile(model) 
    
    # Loss with weights
    criterion_c = nn.BCEWithLogitsLoss(pos_weight=pos_weight_c)
    criterion_s = nn.BCEWithLogitsLoss(pos_weight=pos_weight_s)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    # Training Loop
    print("Starting training...")
    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            token_feats = batch['token_features'].to(device)
            mlp_feats = batch['mlp_features'].to(device)
            target_code = batch['label_code'].unsqueeze(1).to(device)
            target_summary = batch['label_summary'].unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            p_code_logit, p_summ_logit = model(token_feats, mlp_feats)
            
            loss = criterion_c(p_code_logit, target_code) + criterion_s(p_summ_logit, target_summary)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Loss: {avg_loss:.4f}")
        
    # Evaluation
    print("Evaluating...")
    model.eval()
    
    all_code_preds = []
    all_code_targets = []
    all_code_probs = [] # Debugging
    
    all_summ_preds = []
    all_summ_targets = []
    
    trust_scores_model = []
    trust_scores_expert = []
    
    weights = cfg['weights']
    
    with torch.no_grad():
        for batch in test_loader:
            token_feats = batch['token_features'].to(device)
            mlp_feats = batch['mlp_features'].to(device)
            target_code = batch['label_code'].unsqueeze(1).to(device)
            target_summary = batch['label_summary'].unsqueeze(1).to(device)
            s_api = batch['s_api'].to(device)
            
            p_code_logit, p_summ_logit = model(token_feats, mlp_feats)
            
            # Probs
            p_code = torch.sigmoid(p_code_logit)
            p_summ = torch.sigmoid(p_summ_logit)
            
            # Binarize
            preds_c = (p_code > 0.5).float()
            preds_s = (p_summ > 0.5).float()
            
            all_code_preds.extend(preds_c.cpu().numpy())
            all_code_targets.extend(target_code.cpu().numpy())
            all_code_probs.extend(p_code.cpu().numpy())
            
            all_summ_preds.extend(preds_s.cpu().numpy())
            all_summ_targets.extend(target_summary.cpu().numpy())
            
            # Trust Scores
            max_hallucination = torch.max(target_code, target_summary).squeeze()
            expert_score = 1.0 - max_hallucination
            
            model_score = calculate_trust_score(
                p_code.squeeze(), 
                p_summ.squeeze(), 
                s_api, 
                w1=weights['w1'], 
                w2=weights['w2'], 
                w3=weights['w3']
            )
            
            # Handle batch size 1 case for extend
            if expert_score.ndim == 0:
                trust_scores_expert.append(expert_score.item())
                trust_scores_model.append(model_score.item())
            else:
                trust_scores_expert.extend(expert_score.cpu().numpy())
                trust_scores_model.extend(model_score.cpu().numpy())
            
    # Metrics
    prec_c = precision_score(all_code_targets, all_code_preds, zero_division=0)
    rec_c = recall_score(all_code_targets, all_code_preds, zero_division=0)
    f1_c = f1_score(all_code_targets, all_code_preds, zero_division=0)
    
    prec_s = precision_score(all_summ_targets, all_summ_preds, zero_division=0)
    rec_s = recall_score(all_summ_targets, all_summ_preds, zero_division=0)
    f1_s = f1_score(all_summ_targets, all_summ_preds, zero_division=0)
    
    # Correlation
    try:
        corr, _ = spearmanr(trust_scores_expert, trust_scores_model)
    except:
        corr = 0.0
    
    print("\n=== Metrics ===")
    print(f"Code Hallucination: P={prec_c:.3f}, R={rec_c:.3f}, F1={f1_c:.3f}")
    
    # Debugging Code Head
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_code_targets, all_code_preds, labels=[0, 1])
    print(f"Code Confusion Matrix (TN, FP, FN, TP):\n{cm}")
    
    # Prob stats
    probs = np.array(all_code_probs)
    targets = np.array(all_code_targets).flatten()
    print(f"Code Prob Stats: Mean={probs.mean():.3f}, Std={probs.std():.3f}, Min={probs.min():.3f}, Max={probs.max():.3f}")
    if len(targets) > 0:
        print(f"Avg Prob for Positives (Hal): {probs[targets==1].mean() if np.any(targets==1) else 0:.3f}")
        print(f"Avg Prob for Negatives (Ok): {probs[targets==0].mean() if np.any(targets==0) else 0:.3f}")

    print(f"Summary Hallucination: P={prec_s:.3f}, R={rec_s:.3f}, F1={f1_s:.3f}")
    print(f"Trust Score Correlation (Spearman): {corr:.3f}")

    with open('results.txt', 'w') as f:
        f.write(f"Code Hallucination: P={prec_c:.3f}, R={rec_c:.3f}, F1={f1_c:.3f}\n")
        f.write(f"Code Confusion Matrix:\n{cm}\n")
        f.write(f"Summary Hallucination: P={prec_s:.3f}, R={rec_s:.3f}, F1={f1_s:.3f}\n")
        f.write(f"Trust Score Correlation: {corr:.3f}\n")

if __name__ == "__main__":
    train()
