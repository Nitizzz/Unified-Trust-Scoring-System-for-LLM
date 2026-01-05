import pandas as pd
import torch
import features
import ast
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np

PROCESSED_DATA_PATH = 'processed_trust_dataset.parquet'

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    return text.strip()

def check_syntax(code):
    try:
        ast.parse(code)
        return True
    except:
        return False

def load_and_process_data(filepath='code/fyp dataset.xlsx'):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath, sheet_name='DetailedTrustDataset')
    except Exception as e:
        print(f"Error loading excel: {e}")
        return
        
    processed_rows = []
    
    for idx, row in df.iterrows():
        q_id = row['id']
        question = preprocess_text(row['question'])
        orig_code = preprocess_text(row['code'])
        orig_summary = preprocess_text(row['summary'])
        
        if not check_syntax(orig_code):
            print(f"Skipping row {q_id} due to syntax error in original code.")
            continue
            
        # 1. Faithful Variant (Ground Truth)
        # We can add slight noise if needed, but for now exact copy.
        variants = [
            {
                'variant_type': 'faithful',
                'code': orig_code,
                'summary': orig_summary,
                'label_code': 0.0,
                'label_summary': 0.0
            }
        ]
        
        # 2. Hallucinated Code Variant
        mut_code, label_c = features.mutate_code(orig_code)
        if label_c == 1:
            variants.append({
                'variant_type': 'hallucinated_code',
                'code': mut_code,
                'summary': orig_summary,
                'label_code': 1.0, 
                'label_summary': 0.0
            })
            
        # 3. Hallucinated Summary Variant
        mut_summary, label_s = features.mutate_summary(orig_summary, orig_code)
        if label_s == 1:
            variants.append({
                'variant_type': 'hallucinated_summary',
                'code': orig_code,
                'summary': mut_summary,
                'label_code': 0.0,
                'label_summary': 1.0
            })
            
        # Feature Extraction for all variants
        for v_idx, var in enumerate(variants):
            # Token Features
            token_feats = features.tokenize_code(var['code']) # Tensor
            
            # Execution Features
            # We don't have explicit tests in the excel, using dummy success check
            exec_feats = features.execute_and_score(var['code'], [])
            
            # Summary Features
            summ_feats = features.analyze_summary_faithfulness(var['code'], var['summary'])
            
            # API Features
            api_feats = features.check_api_usage(var['code'])
            
            processed_rows.append({
                'id': q_id,
                'variant_id': f"{q_id}_{v_idx}",
                'question': question,
                'code': var['code'],
                'summary': var['summary'],
                'token_features': token_feats.tolist(), # Store as list for parquet
                'exec_features': exec_feats.tolist(),
                'summary_features': summ_feats.tolist(),
                'api_features': api_feats.tolist(),
                'label_code': var['label_code'],
                'label_summary': var['label_summary'],
                's_api': api_feats[1].item() # Extract score
            })
            
    # Save to parquet
    result_df = pd.DataFrame(processed_rows)
    # Ensure lists are stored as strings or handle them. Parquet handles arrays but pandas-parquet mapping can be tricky with pyarrow.
    # Simple workaround: just pickle or use json serialization for complex columns if parquet fails, 
    # but let's try direct parquet (pyarrow 2.0+ supports complex types).
    try:
        result_df.to_parquet(PROCESSED_DATA_PATH, engine='pyarrow')
        print(f"Saved processed dataset to {PROCESSED_DATA_PATH} with {len(result_df)} samples.")
    except Exception as e:
        print(f"Parquet save failed: {e}. Falling back to pickle.")
        result_df.to_pickle(PROCESSED_DATA_PATH.replace('.parquet', '.pkl'))

class TrustDataset(Dataset):
    def __init__(self, data_path=PROCESSED_DATA_PATH):
        if data_path.endswith('.parquet'):
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_pickle(data_path)
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Helper to ensure list
        def to_list(x):
            if hasattr(x, 'tolist'): return x.tolist()
            return x
            
        # Convert lists back to tensors (Optimized)
        token_feats_np = np.array(to_list(row['token_features']), dtype=np.float32)
        exec_feats_np = np.array(to_list(row['exec_features']), dtype=np.float32)
        summ_feats_np = np.array(to_list(row['summary_features']), dtype=np.float32)
        api_feats_np = np.array(to_list(row['api_features']), dtype=np.float32)
        
        token_feats = torch.from_numpy(token_feats_np)
        exec_feats = torch.from_numpy(exec_feats_np)
        summ_feats = torch.from_numpy(summ_feats_np)
        api_feats = torch.from_numpy(api_feats_np)
        
        # MLP features concatenation: Exec + Summary + API
        # Dimensions: Exec (12), Summary (3), API (3) -> Total ~18
        mlp_feats = torch.cat([exec_feats, summ_feats, api_feats], dim=0)
        
        return {
            'token_features': token_feats,
            'mlp_features': mlp_feats,
            'label_code': torch.tensor(float(row['label_code']), dtype=torch.float32),
            'label_summary': torch.tensor(float(row['label_summary']), dtype=torch.float32),
            's_api': torch.tensor(float(row['s_api']), dtype=torch.float32)
        }

if __name__ == "__main__":
    load_and_process_data()
