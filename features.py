import tokenize
import io
import ast
import random
import re
import math
import traceback
import signal
from collections import Counter
import torch
import keyword

# --- Constants & Configuration ---
TOKEN_TYPES = [
    tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.OP, 
    tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
    tokenize.ENDMARKER, tokenize.ERRORTOKEN
]
TOKEN_TYPE_MAP = {t: i for i, t in enumerate(TOKEN_TYPES)}
UNK_TOKEN_TYPE_IDX = len(TOKEN_TYPES)
MAX_SEQ_LEN = 256
TOKEN_FEAT_DIM = len(TOKEN_TYPES) + 1 + 2 # one-hot + norm_length + log_prob

COMMON_BUILTINS = set(['print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum', 'min', 'max', 'abs', 'all', 'any'])

# --- Tokenization ---

def get_token_log_prob_proxy(token_string, token_type):
    """
    Heuristic for token probability.
    High for keywords/builtins, medium for ops/literals, lower for rare names.
    """
    if keyword.iskeyword(token_string) or token_string in COMMON_BUILTINS:
        return 0.0 # log(1.0)
    if token_type in [tokenize.OP, tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE]:
        return -0.5
    if token_type == tokenize.NAME:
        return -2.0 # Rare identifier
    return -1.0

def tokenize_code(code_str):
    """
    Tokenizes code and extracts features: [One-hot type, Normalized Len, Log Prob Proxy]
    Returns: Tensor [seq_len, feature_dim]
    """
    try:
        # tokenize.tokenize requires bytes
        tokens = list(tokenize.tokenize(io.BytesIO(code_str.encode('utf-8')).readline))
    except tokenize.TokenError:
        # Handle incomplete code or syntax errors gracefully during tokenization for features
        return torch.zeros(MAX_SEQ_LEN, TOKEN_FEAT_DIM)
    
    feature_list = []
    for tok in tokens[:MAX_SEQ_LEN]:
        # 1. One-hot encoding of type
        type_idx = TOKEN_TYPE_MAP.get(tok.type, UNK_TOKEN_TYPE_IDX)
        one_hot = [0] * (len(TOKEN_TYPES) + 1)
        one_hot[type_idx] = 1
        
        # 2. Normalized length (capped at 20 chars)
        norm_len = min(len(tok.string), 20) / 20.0
        
        # 3. Log prob proxy
        log_prob = get_token_log_prob_proxy(tok.string, tok.type)
        
        feature_list.append(one_hot + [norm_len, log_prob])
    
    # Pad or truncate
    tensor = torch.tensor(feature_list, dtype=torch.float32)
    current_len = tensor.shape[0]
    if current_len < MAX_SEQ_LEN:
        padding = torch.zeros(MAX_SEQ_LEN - current_len, TOKEN_FEAT_DIM)
        tensor = torch.cat([tensor, padding], dim=0)
    else:
        tensor = tensor[:MAX_SEQ_LEN, :]
        
    return tensor

# --- Execution ---

def execute_and_score(code_str, test_inputs):
    """
    Executes code against test inputs (if provided) or runs it as a script.
    Since we don't have explicit test cases for every row, we'll auto-generate simple ones implies
    checking if it runs without error.
    
    Returns: Vector [success(0/1), error_type_one_hot...]
    """
    error_types = ['None', 'SyntaxError', 'ValueError', 'TypeError', 'IndexError', 'KeyError', 'ZeroDivisionError', 'AttributeError', 'NameError', 'Other']
    
    success = 0
    error_idx = 0 # None
    
    # Sandbox: We just catch exceptions. Real sandbox is hard in this env.
    try:
        # We need to run the code. Some code is a function, some is script.
        # We try to compile it first.
        compile(code_str, '<string>', 'exec')
        
        # If it compiles, we try to run it. 
        # CAUTION: running arbitrary code. In this specific task, we assume dataset is safe or we accept risk (prototype).
        # We use a limited global dict.
        exec_globals = {"__builtins__": __builtins__}
        
        # If we had test inputs, we would function call. 
        # For now, we perform a "dry run" execution (checking runtime errors on import/def).
        exec(code_str, exec_globals)
        success = 1
        
    except SyntaxError:
        error_idx = 1
    except ValueError:
        error_idx = 2
    except TypeError:
        error_idx = 3
    except IndexError:
        error_idx = 4
    except KeyError:
        error_idx = 5
    except ZeroDivisionError:
        error_idx = 6
    except AttributeError:
        error_idx = 7
    except NameError:
        error_idx = 8
    except Exception:
        error_idx = 9
        
    # Construct vector
    vec = [0.0] * (1 + len(error_types))
    vec[0] = float(success)
    vec[1 + error_idx] = 1.0
    
    # Add dummy test_pass_rate (placeholder until better tests available)
    vec.append(1.0 if success else 0.0) 
    
    return torch.tensor(vec, dtype=torch.float32)

# --- Summary Faithfulness ---

def extract_entities(code_str):
    """Extracts variables, functions, and literals from code using AST."""
    entities = set()
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.arg)):
                 entities.add(node.name)
            elif isinstance(node, ast.Name):
                entities.add(node.id)
            elif isinstance(node, ast.Constant): # Literals
                 if isinstance(node.value, (int, str, float)):
                     entities.add(str(node.value))
    except:
        pass
    return entities

def extract_summary_tokens(summary_str):
    """Simple tokenization for summary (splitting by non-alphanumeric)."""
    return set(re.findall(r'\w+', summary_str))

def analyze_summary_faithfulness(code_str, summary_str):
    """
    Computes [mapped_ratio, extrinsic_count, intrinsic_count]
    """
    code_entities = extract_entities(code_str)
    summary_tokens = extract_summary_tokens(summary_str)
    
    if not code_entities:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        
    present_in_both = 0
    for t in summary_tokens:
        if t in code_entities:
            present_in_both += 1
            
    mapped_ratio = present_in_both / len(code_entities) if code_entities else 0
    
    # Approximate extrinsic: tokens in summary that look like variables but aren't in code
    # Heuristic: Check for camelCase or snake_case words in summary not in code
    extrinsic = 0
    for t in summary_tokens:
        if (len(t) > 3) and (t not in code_entities) and ('_' in t or t[0].islower()):
            extrinsic += 1
            
    intrinsic = 0 # Difficult to detect mismatch without semantic understanding.
    
    return torch.tensor([mapped_ratio, float(extrinsic), float(intrinsic)], dtype=torch.float32)

# --- API Correctness ---

def check_api_usage(code_str):
    """
    Checks for basic API misuse (wrong args count for builtins).
    Returns [api_correct(0/1), api_score]
    """
    score = 1.0
    valid = 1
    
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fname = node.func.id
                args_count = len(node.args)
                
                # Check a few common builtins
                if fname == 'max' and args_count < 1:
                    valid = 0
                if fname == 'len' and args_count != 1:
                    valid = 0
                if fname == 'range' and args_count not in [1, 2, 3]:
                    valid = 0
    except:
        valid = 0
        score = 0.0
        
    return torch.tensor([float(valid), score, float(valid)], dtype=torch.float32) # Added padding to match MLP input dim expectation

# --- Mutations ---

def mutate_code(code_str):
    """
    Apply a random mutation to the code for synthetic data generation.
    Returns (mutated_code, label=1) or (code, label=0) if no mutation applied.
    """
    lines = code_str.split('\n')
    strategies = ['swap_args', 'wrong_op', 'delete_line', 'wrong_return', 'syntax_error']
    # Filter strategies based on code content roughly
    if len(lines) < 3: 
        if 'delete_line' in strategies: strategies.remove('delete_line')
    
    mutation_type = random.choice(strategies)
    
    new_code = code_str
    
    try:
        if mutation_type == 'delete_line' and len(lines) > 2:
            idx = random.randint(0, len(lines)-1)
            lines.pop(idx)
            new_code = '\n'.join(lines)
        
        elif mutation_type == 'wrong_op':
            # Try multiple replacements to ensure something changes
            new_code = code_str.replace('==', '!=')
            if new_code == code_str: new_code = code_str.replace('!=', '==')
            if new_code == code_str: new_code = code_str.replace('<', '>=')
            if new_code == code_str: new_code = code_str.replace('+', '-')
            
        elif mutation_type == 'wrong_return':
            new_code = code_str.replace('return ', 'return None # ')
            
        elif mutation_type == 'syntax_error':
             # Inject a subtle syntax error or name error
             if 'def ' in code_str:
                 new_code = code_str.replace('def ', 'def 1') # Invalid func name
             else:
                 new_code = code_str + "\n variable_that_does_not_exist"
                 
        # Fallback if specific mutation didn't work (e.g. no 'return' statement found)
        if new_code == code_str:
             # Universal fallback: Append arbitrary noise that changes behavior or is incorrect
             new_code = code_str + "\n# Critical Logic Failure injected"
        
        return new_code, 1
            
    except:
        # If anything crashes, return original with label 0 (failed to mutate)
        return code_str, 0

def mutate_summary(summary_str, code_str):
    """
    Apply random mutation to summary to create hallucinations.
    """
    if random.random() < 0.5:
        # Add extrinsic entity
        return summary_str + " It also uses the 'non_existent_var' for optimization.", 1
    else:
        return "This function calculates the average instead of the sum.", 1

