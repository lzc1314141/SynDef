import os
import re
import sys
import csv
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
from functools import lru_cache

csv.field_size_limit(min(2147483647, sys.maxsize))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Tip: install tqdm for better progress display (pip install tqdm)")

sys.path.insert(0, '/root/workspace/lzc/SynDef')
from transform import JavaCodeTokenizer
from automat import SyntaxAutomaton, ParseState

# ------------------------------------------------
#  (A = (Q, Σ, δ, q0))
# ------------------------------------------------
class SyntaxStackAutomaton:
    
    def __init__(self, vocabulary_automaton: SyntaxAutomaton):
        
        self.sa = vocabulary_automaton          # Contains predict_legal_tokens() mapping
        self.current_stack = ['root']          # Initial state q0

    def reset(self):
        self.current_stack = ['root']

    def step_enter(self, node_type: str):
        self.current_stack.append(node_type)

    def step_exit(self):
        if len(self.current_stack) > 1:       
            self.current_stack.pop()

    def get_current_state(self) -> ParseState:
        return ParseState(list(self.current_stack))

    def get_legal_mask(self, parse_state: ParseState, device='cpu'):
        mask_np = self.sa.predict_legal_tokens(parse_state)
        return torch.from_numpy(mask_np).to(device)


CONTROL_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
    'try', 'catch', 'finally', 'synchronized', 'foreach', 'for_each',
    'lambda', 'break', 'continue'
}
CONTROL_KEYWORDS_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in sorted(CONTROL_KEYWORDS)) + r')\b',
    re.IGNORECASE
)

EXCEPTION_RESOURCE_PATTERN = re.compile(
    r'\b(?:try|catch|finally|throw|throws|AutoCloseable|close\s*\(|InputStream|OutputStream|Reader|Writer|SQLException|IOException)\b',
    re.IGNORECASE
)

CONCURRENCY_PATTERN = re.compile(
    r'\b(?:synchronized|volatile|ReentrantLock|ReadWriteLock|Semaphore|CountDownLatch|ThreadLocal|CompletableFuture|ExecutorService|Atomic\w+|Thread)\b',
    re.IGNORECASE
)

SENSITIVE_API_PATTERN = re.compile(
    r'\b(?:Socket|ServerSocket|HttpClient|URLConnection|Class\.forName|Method\.invoke|Field\.set|Runtime\.getRuntime|ProcessBuilder|System\.loadLibrary|JNIEnv)\b',
    re.IGNORECASE
)

def detect_control_keyword(line_text):
    if not line_text: return 0
    return 1 if CONTROL_KEYWORDS_PATTERN.search(line_text) else 0

def detect_exception_resource(line_text):
    if not line_text: return 0
    return 1 if EXCEPTION_RESOURCE_PATTERN.search(line_text) else 0

def detect_concurrency_sync(line_text):
    if not line_text: return 0
    return 1 if CONCURRENCY_PATTERN.search(line_text) else 0

def detect_sensitive_api(line_text):
    if not line_text: return 0
    return 1 if SENSITIVE_API_PATTERN.search(line_text) else 0

def detect_code_smell(line_text):
    if not line_text: return 0
    stripped = line_text.strip()
    if not stripped: return 0
    long_line = len(stripped) > 120
    dense_boolean = stripped.count('&&') + stripped.count('||') >= 2
    chained_calls = stripped.count('.') >= 5 and '(' in stripped
    todo_comment = any(tag in stripped for tag in ('TODO', 'FIXME', 'HACK'))
    ternary_complex = stripped.count('?') >= 1 and stripped.count(':') >= 1
    return 1 if (long_line or dense_boolean or chained_calls or todo_comment or ternary_complex) else 0

def count_defect_elements(line_text):
    return (detect_control_keyword(line_text) +
            detect_exception_resource(line_text) +
            detect_concurrency_sync(line_text) +
            detect_sensitive_api(line_text) +
            detect_code_smell(line_text))

COMMENT_PATTERN_1 = re.compile(r'//.*$', re.MULTILINE)
COMMENT_PATTERN_2 = re.compile(r'/\*.*?\*/', re.DOTALL)
TOKEN_PATTERN = re.compile(r'\b\w+\b|[+\-*/=<>!&|]+|[(){}\[\];,.]|"[^"]*"|\'[^\']*\'')

class UltraFileCache:
    def __init__(self, max_cache_size=400):
        self.cache = {}
        self.access_count = {}
        self.max_cache_size = max_cache_size
    def _evict_if_needed(self):
        if len(self.cache) < self.max_cache_size: return
        lru_key = min(self.access_count, key=self.access_count.get)
        self.cache.pop(lru_key, None)
        self.access_count.pop(lru_key, None)
    def get_file_content(self, file_path):
        if file_path not in self.cache:
            if not os.path.exists(file_path): return None
            self._evict_if_needed()
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.cache[file_path] = f.readlines()
            self.access_count[file_path] = 0
        self.access_count[file_path] += 1
        return self.cache[file_path]
    def get_line(self, file_path, line_number):
        lines = self.get_file_content(file_path)
        if lines and 1 <= line_number <= len(lines):
            return lines[line_number - 1].rstrip('\n\r')
        return None

class UltraAutomataProcessor:
    def __init__(self, automata_data):
        self.automata_list = []
        self._preprocess_automata(automata_data)
    def _preprocess_automata(self, automata_data):
        for automaton_entry in automata_data.get('automata', []):
            automaton = automaton_entry.get('automaton', {})
            if not automaton: continue
            processed = {
                'start_state': automaton.get('start_state', 'q0'),
                'accept_states': frozenset(automaton.get('accept_states', [])),
                'transition_map': {},
                'first_symbols': set(),
                'required_symbols': set(),
            }
            for trans in automaton.get('transitions', []):
                from_state = trans.get('from_state')
                to_state = trans.get('to_state')
                symbol = trans.get('symbol')
                if not from_state or not to_state or symbol is None: continue
                if from_state == processed['start_state']:
                    processed['first_symbols'].add(symbol)
                processed['required_symbols'].add(symbol)
                processed['transition_map'].setdefault(from_state, {})[symbol] = to_state
            if processed['accept_states']:
                self.automata_list.append(processed)
    def check_tokens_fast(self, input_tokens):
        if not input_tokens: return False
        token_set = frozenset(input_tokens)
        for automaton in self.automata_list:
            if not automaton['required_symbols'].issubset(token_set): continue
            if input_tokens[0] not in automaton['first_symbols']: continue
            if self._can_reach_final_state(automaton, input_tokens): return True
        return False
    @staticmethod
    def _can_reach_final_state(automaton, input_tokens):
        current_state = automaton['start_state']
        transition_map = automaton['transition_map']
        for token in input_tokens:
            if current_state not in transition_map: return False
            if token not in transition_map[current_state]: return False
            current_state = transition_map[current_state][token]
        return current_state in automaton['accept_states']

@lru_cache(maxsize=50000)
def tokenize_java_line_ultra_cached(line_content):
    if not line_content or not line_content.strip(): return tuple()
    line = COMMENT_PATTERN_1.sub('', line_content)
    line = COMMENT_PATTERN_2.sub('', line)
    if not line.strip(): return tuple()
    tokens = TOKEN_PATTERN.findall(line)
    return tuple(token.strip() for token in tokens if token.strip())

def convert_csv_path_to_file_name(csv_path):
    return csv_path.replace('.java', '').replace('/', '_').replace('\\', '_') + '.java'

def scale_to_range(value, src_min, src_max, dst_min, dst_max):
    if value is None or src_min is None or src_max is None: return None
    if src_max == src_min: return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    ratio = max(0.0, min(1.0, ratio))
    return dst_min + ratio * (dst_max - dst_min)

def safe_float(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default

def compute_rank_bucket(line_surprisal, optimized_score):
    if line_surprisal != 0.0: return 0
    if 1.0 <= optimized_score <= 3.0: return 1
    if 0.5 < optimized_score < 1.0: return 2
    if 0.0 < optimized_score <= 0.5: return 3
    return 4

def get_predefined_version_mapping():
    return {
        'activemq-5.1.0': ['activemq-5.0.0'],
        'activemq-5.2.0': ['activemq-5.0.0', 'activemq-5.1.0'],
        'activemq-5.3.0': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0'],
        'activemq-5.8.0': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0'],
        'camel-2.10.0': ['camel-1.4.0', 'camel-2.9.0'],
        'camel-2.11.0': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0'],
        'derby-10.5.1.1': ['derby-10.2.1.6', 'derby-10.3.1.4'],
        'groovy-1_6_BETA_2': ['groovy-1_5_7', 'groovy-1_6_BETA_1'],
        'hbase-0.95.2': ['hbase-0.94.0', 'hbase-0.95.0'],
        'hive-0.12.0': ['hive-0.9.0', 'hive-0.10.0'],
        'jruby-1.5.0': ['jruby-1.1', 'jruby-1.4.0'],
        'jruby-1.7.0.preview1': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0'],
        'lucene-3.0.0': ['lucene-2.3.0', 'lucene-2.9.0'],
        'lucene-3.1': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0'],
        'wicket-1.5.3': ['wicket-1.3.0-beta2', 'wicket-1.3.0-incubating-beta-1'],
    }

OPTIMIZED_SCORE_WEIGHTS = {
    "line_entropy_z": 1.848,
    "line_surprisal_z": -2.043,
    "orig_entropy_z": 0.946,
    "illegal_ratio_z": 0.003,
    "defect_flag": 0.508,
    "defect_count_z": 0.195,
}
OPTIMIZED_SCORE_LOWER = 1.0
OPTIMIZED_SCORE_UPPER = 3.0

def robust_zscore(values):
    arr = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(arr, dtype=np.float32)
    if arr.size == 0: return result
    finite_mask = np.isfinite(arr)
    if not finite_mask.any(): return result
    valid = arr[finite_mask]
    median = float(np.nanmedian(valid))
    q1 = float(np.nanpercentile(valid, 25))
    q3 = float(np.nanpercentile(valid, 75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr < 1e-6:
        std = float(np.nanstd(valid))
        scale = std if np.isfinite(std) and std >= 1e-6 else 1.0
    else:
        scale = iqr
    result[finite_mask] = (arr[finite_mask] - median) / scale
    return result

def compute_optimized_features(rows):
    def arr(field, default=0.0):
        return np.array([float(row.get(field, default)) for row in rows], dtype=np.float32)
    line_entropy = arr('line_entropy', 0.0)
    line_surprisal = arr('line_surprisal', 0.0)
    orig_entropy = arr('orig_entropy', 0.0)
    illegal_token_count = arr('illegal_token_count', 0.0)
    defect_counts = arr('defect_element_count', 0.0)
    token_count = np.maximum(arr('token_count', 1.0), 1.0)
    return {
        "line_entropy_z": robust_zscore(line_entropy),
        "line_surprisal_z": robust_zscore(line_surprisal),
        "orig_entropy_z": robust_zscore(orig_entropy),
        "illegal_ratio_z": robust_zscore(illegal_token_count / token_count),
        "defect_flag": (defect_counts > 0).astype(np.float32),
        "defect_count_z": robust_zscore(defect_counts),
    }

def compute_optimized_scores(rows):
    if not rows: return np.array([], dtype=np.float32)
    features = compute_optimized_features(rows)
    return sum(OPTIMIZED_SCORE_WEIGHTS[key] * features[key] for key in OPTIMIZED_SCORE_WEIGHTS)

def compress_optimized_scores(length):
    if length <= 0: return np.array([], dtype=np.float32)
    if length == 1: return np.array([OPTIMIZED_SCORE_UPPER], dtype=np.float32)
    span = OPTIMIZED_SCORE_UPPER - OPTIMIZED_SCORE_LOWER
    last = length - 1
    return np.array(
        [OPTIMIZED_SCORE_UPPER - span * (idx / last) for idx in range(length)],
        dtype=np.float32,
    )

def calculate_entropy(probs):
    probs = probs + 1e-10
    return -torch.sum(probs * torch.log(probs)).item()

def load_n_gram_result_lines(n_gram_result_file_path):
    file_lines_map = defaultdict(set)
    if not os.path.exists(n_gram_result_file_path):
        print(f"  Warning: n_gram_result file not found: {n_gram_result_file_path}")
        return file_lines_map
    try:
        with open(n_gram_result_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.readline()  
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    file_path = parts[2].strip()
                    try:
                        line_number = int(parts[3].strip())
                        if file_path and line_number > 0:
                            file_lines_map[file_path].add(line_number)
                    except (ValueError, IndexError): continue
        print(f"  Read line number info for {len(file_lines_map)} files from n_gram_result file")
        total_lines = sum(len(v) for v in file_lines_map.values())
        print(f"  Total {total_lines} lines to compute")
    except Exception as e:
        print(f"  Error: failed to read n_gram_result file: {e}")
        import traceback; traceback.print_exc()
    return file_lines_map


def compute_parse_states_by_stack(tree, tokens, stack_automaton: SyntaxStackAutomaton):
    
    states = [None] * len(tokens)
    stack_automaton.reset()          
    token_idx = 0

    def assign_state_to_token(node):
        nonlocal token_idx
        if not hasattr(node, 'position') or node.position is None: return
        while token_idx < len(tokens):
            tok_str, tok_line, tok_col = tokens[token_idx]
            if tok_line == node.position.line and tok_col == node.position.column:
                node_val = getattr(node, 'value', None)
                if node_val is None or node_val == tok_str:
                    states[token_idx] = stack_automaton.get_current_state()
                    token_idx += 1
                    return
            token_idx += 1

    def dfs(node):
        node_type = type(node).__name__
        stack_automaton.step_enter(node_type)   

        assign_state_to_token(node)

        for child_name, child in node.children:
            if isinstance(child, list):
                for item in child:
                    if hasattr(item, 'children'): dfs(item)
                    else: assign_state_to_token(item)
            elif hasattr(child, 'children'): dfs(child)
            else: assign_state_to_token(child)

        stack_automaton.step_exit()             

    dfs(tree)

    
    for i in range(len(states)):
        if states[i] is None:
            states[i] = ParseState(['root'])
    return states


def calculate_line_entropy(model, tokenizer, code_file_path, device='cuda',
                          max_seq_length=None, target_line_numbers=None,
                          automaton: SyntaxAutomaton = None):
    model.eval()
    model.to(device)

    if max_seq_length is None:
        if hasattr(model, 'max_seq_length'): max_length = model.max_seq_length
        elif hasattr(model, 'positional_encoding') and hasattr(model.positional_encoding, 'max_seq_length'):
            max_length = model.positional_encoding.max_seq_length
        else: max_length = 512
    else: max_length = max_seq_length

    with open(code_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    file_name = os.path.basename(code_file_path)
    total_lines = len(lines)
    score_total_lines = len(target_line_numbers) if target_line_numbers else total_lines

    # Simple progress bar
    simple_progress_last_line = 0
    score_simple_last_line = 0
    def _update_simple_progress(cur):
        nonlocal simple_progress_last_line
        if total_lines == 0 or cur == simple_progress_last_line: return
        simple_progress_last_line = cur
        bar_len = 30
        ratio = cur / total_lines
        bar = '#' * int(bar_len*ratio) + '-' * (bar_len - int(bar_len*ratio))
        end = '\r' if cur < total_lines else '\n'
        sys.stdout.write(f"\r  {file_name} line progress [{bar}] {cur}/{total_lines}")
        sys.stdout.flush()
        if end == '\n': sys.stdout.write('\n')
    def _update_score_progress(cnt):
        nonlocal score_simple_last_line
        if score_total_lines == 0 or cnt == score_simple_last_line: return
        score_simple_last_line = cnt
        bar_len = 30
        ratio = cnt / score_total_lines
        bar = '#' * int(bar_len*ratio) + '-' * (bar_len - int(bar_len*ratio))
        sys.stdout.write(f"\r  {file_name} scoring progress [{bar}] {cnt}/{score_total_lines}")
        sys.stdout.flush()
        if cnt >= score_total_lines: sys.stdout.write('\n')

    if HAS_TQDM and total_lines > 0:
        progress_bar = tqdm(total=total_lines, desc=f"{file_name} line processing", unit="lines", leave=False)
    else: progress_bar = None
    if HAS_TQDM and score_total_lines > 0:
        score_progress_bar = tqdm(total=score_total_lines, desc=f"{file_name} line scoring", unit="lines", leave=False)
    else: score_progress_bar = None

    line_entropies = []
    USE_DYNAMIC_STATE = automaton is not None
    USE_LOGITS_HARD_MASK = True
    unk_token_id = getattr(tokenizer, 'unk_token_id', None)
    state_mask_cache = {}

    accumulated_lines = []
    MAX_CONTEXT_LINES = 1000
    context_token_ids = [tokenizer.cls_token_id]
    context_line_token_lengths = deque()

    def append_context(line_token_ids, line_text):
        nonlocal context_token_ids
        line_len = len(line_token_ids) if line_token_ids else 0
        if line_len > 0:
            context_token_ids = context_token_ids + line_token_ids
        context_line_token_lengths.append(line_len)
        accumulated_lines.append(line_text)
        while len(context_line_token_lengths) > MAX_CONTEXT_LINES:
            drop = context_line_token_lengths.popleft()
            if drop > 0 and len(context_token_ids) > 1:
                context_token_ids = [context_token_ids[0]] + context_token_ids[1+drop:]
        while len(accumulated_lines) > MAX_CONTEXT_LINES:
            accumulated_lines.pop(0)

    stack_auto = SyntaxStackAutomaton(automaton) if USE_DYNAMIC_STATE else None

    scored_line_count = 0
    for line_num, line in enumerate(lines, start=1):
        if progress_bar: progress_bar.update(1)
        else: _update_simple_progress(line_num)

        tokens = tokenizer.tokenize_java_code(line)
        line_token_ids = []
        if tokens:
            for tok in tokens:
                line_token_ids.append(tokenizer.token_to_id.get(tok, tokenizer.unk_token_id))

        if target_line_numbers and line_num not in target_line_numbers:
            append_context(line_token_ids, line)
            continue
        if not tokens:
            append_context(line_token_ids, line)
            continue
        if not line_token_ids:
            append_context(line_token_ids, line)
            continue

        input_ids = context_token_ids + line_token_ids
        input_context_token_ids = context_token_ids
        line_len = len(line_token_ids)
        if len(input_ids) > max_length:
            avail = max_length - line_len - 1
            if avail > 0:
                input_context_token_ids = [tokenizer.cls_token_id] + context_token_ids[1:][-avail:]
                input_ids = input_context_token_ids + line_token_ids
            else:
                input_ids = [tokenizer.cls_token_id] + line_token_ids[-max_length+1:]
                input_context_token_ids = [tokenizer.cls_token_id]

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        line_states = None
        if USE_DYNAMIC_STATE:
            try:
                code_so_far = ''.join(accumulated_lines + [line])
                at_tokens = automaton.tokenize_java_code(code_so_far)
                tree = automaton.parse_java_code(code_so_far)
                
                all_states = compute_parse_states_by_stack(tree, at_tokens, stack_auto)
                current_line_states = []
                for i, (tok, ln, col) in enumerate(at_tokens):
                    if ln == line_num:
                        ps = all_states[i] if i < len(all_states) else ParseState(['root'])
                        current_line_states.append(ps)
                if not current_line_states:
                    current_line_states = [ParseState(['root'])] * len(line_token_ids)
                line_states = current_line_states
            except Exception:
                line_states = [ParseState(['root'])] * len(line_token_ids)

        try:
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                token_entropies, token_surprisals, token_orig_entropies = [], [], []
                illegal_token_count = 0
                line_start_pos = len(input_context_token_ids)
                actual_line_tokens_in_input = len(input_ids) - line_start_pos

                for line_token_idx in range(actual_line_tokens_in_input):
                    pos_in_input = line_start_pos + line_token_idx
                    logits_pos = pos_in_input - 1
                    if logits_pos < 0 or logits_pos >= logits.shape[1]: continue

                    token_logits = logits[0, logits_pos, :]
                    if USE_DYNAMIC_STATE and line_states and len(line_states) > 0:
                        state_idx = min(line_token_idx, len(line_states)-1)
                        parse_state = line_states[state_idx]
                    else:
                        parse_state = ParseState(['root'])

                    if automaton and automaton.vocab_size == token_logits.shape[-1] and USE_LOGITS_HARD_MASK:
                        state_id = parse_state.get_state_id()
                        if state_id in state_mask_cache:
                            legal_mask = state_mask_cache[state_id]
                        else:
                            legal_mask = stack_auto.get_legal_mask(parse_state, device)
                            state_mask_cache[state_id] = legal_mask

                        if torch.sum(legal_mask).item() == 0:
                            root_id = 'root'
                            if root_id in state_mask_cache:
                                legal_mask = state_mask_cache[root_id]
                            else:
                                root_mask = stack_auto.get_legal_mask(ParseState(['root']), device)
                                state_mask_cache[root_id] = root_mask
                                legal_mask = root_mask

                        legal_bool = (legal_mask != 0)
                        probs_legal = F.softmax(token_logits.masked_fill(~legal_bool, -1e9), dim=-1)
                        probs_illegal = F.softmax(token_logits.masked_fill(legal_bool, -1e9), dim=-1)

                        idx_in_line = len(line_token_ids) - actual_line_tokens_in_input + line_token_idx
                        idx_in_line = max(0, min(idx_in_line, len(line_token_ids)-1))
                        current_token_id = line_token_ids[idx_in_line]

                        is_unk = (unk_token_id is not None and current_token_id == unk_token_id)
                        if is_unk:
                            is_legal = True
                        else:
                            try:
                                is_legal = bool(legal_mask[current_token_id].item() != 0)
                            except Exception:
                                is_legal = False

                        token_entropy = calculate_entropy(probs_legal)
                        probs_default = F.softmax(token_logits, dim=-1)
                        token_orig_entropy = calculate_entropy(probs_default)

                        if is_legal:
                            token_surprisal = 0.0
                        else:
                            illegal_token_count += 1
                            p_tok = probs_illegal[current_token_id].item() if current_token_id < probs_illegal.shape[-1] else 0.0
                            if p_tok <= 0.0: p_tok = 1e-12
                            token_surprisal = float(-np.log(p_tok))
                    else:
                        probs_default = F.softmax(token_logits, dim=-1)
                        token_entropy = calculate_entropy(probs_default)
                        token_orig_entropy = calculate_entropy(probs_default)
                        token_surprisal = 0.0

                    token_entropies.append(token_entropy)
                    token_surprisals.append(token_surprisal)
                    token_orig_entropies.append(token_orig_entropy)

                if token_entropies:
                    line_entropies.append({
                        'line_num': line_num,
                        'line_entropy': float(np.mean(token_entropies)),
                        'line_surprisal': float(np.mean(token_surprisals)) if token_surprisals else 0.0,
                        'defect_element_count': count_defect_elements(line),
                        'orig_entropy': float(np.mean(token_orig_entropies)) if token_orig_entropies else 0.0,
                        'delta_entropy': float(np.mean(token_orig_entropies)) - float(np.mean(token_entropies)),
                        'illegal_token_count': illegal_token_count,
                        'token_count': len(line_token_ids)
                    })
                    scored_line_count += 1
                    if score_progress_bar: score_progress_bar.update(1)
                    else: _update_score_progress(scored_line_count)
        except Exception as e:
            print(f"Warning: error processing file {file_name} line {line_num}: {e}")
            import traceback; traceback.print_exc()

        append_context(line_token_ids, line)

    if progress_bar: progress_bar.close()
    elif total_lines == 0: _update_simple_progress(0)
    if score_progress_bar: score_progress_bar.close()
    elif score_total_lines == 0: _update_score_progress(0)

    return line_entropies

_global_model = None
_global_tokenizer = None
_global_device = None
_global_automaton = None

def init_worker(model_path, tokenizer_path, device):
    global _global_model, _global_tokenizer, _global_device, _global_automaton
    _global_device = device
    from transform import JavaCodeTokenizer, TransformerModel, create_model

    _global_tokenizer = JavaCodeTokenizer()
    _global_tokenizer.load_vocab(tokenizer_path)

    try:
        sa = SyntaxAutomaton(vocabulary_path=tokenizer_path)
        sa.load('/root/workspace/lzc/SynDef/automat-model')
        _global_automaton = sa
    except Exception:
        _global_automaton = None

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model_cfg = checkpoint.get('model_config', None)
    vocab_size = int(checkpoint.get('vocab_size', state_dict['embedding.weight'].shape[0]))
    if model_cfg:
        try:
            d_model = int(model_cfg['d_model'])
            n_heads = int(model_cfg['n_heads'])
            n_layers = int(model_cfg['n_layers'])
            d_ff = int(model_cfg['d_ff'])
            max_seq_length = int(model_cfg['max_seq_length'])
        except Exception:
            raise RuntimeError(f"model_config fields incomplete: {model_cfg}")
    else:
        from inspect import signature
        try:
            sig = signature(create_model)
            default_kw = {k: v.default for k, v in sig.parameters.items() if v.default is not sig.empty}
            default_n_heads = int(default_kw.get('n_heads', 8))
        except Exception:
            default_n_heads = 8
        d_model = int(state_dict['embedding.weight'].shape[1])
        n_layers = len([k for k in state_dict if 'transformer_blocks' in k and 'attention.W_q.weight' in k])
        d_ff = int(state_dict['transformer_blocks.0.feed_forward.0.weight'].shape[0]) if n_layers > 0 else 2048
        max_seq_length = int(state_dict['positional_encoding.pe'].shape[1]) if 'positional_encoding.pe' in state_dict else 512
        for h in [default_n_heads, 4, 8, 12, 16, 24, 32]:
            if h > 0 and d_model % h == 0:
                n_heads = h
                break
        if n_heads is None:
            raise RuntimeError(f"Unable to determine n_heads for d_model={d_model}")
    if d_model <= 0 or n_heads <= 0 or n_layers <= 0 or d_ff <= 0 or max_seq_length <= 0:
        raise RuntimeError("Invalid model parameters")
    if d_model % n_heads != 0:
        raise RuntimeError(f"d_model ({d_model}) is not divisible by n_heads ({n_heads})")

    _global_model = TransformerModel(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, max_seq_length=max_seq_length,
        dropout=0.1, pad_token_id=_global_tokenizer.pad_token_id
    )
    _global_model.load_state_dict(state_dict)
    _global_model.eval()
    _global_model.to(device)

def process_single_file(args):
    java_file_path, file_name, max_seq_length, target_line_numbers = args
    try:
        if target_line_numbers is None or len(target_line_numbers) == 0:
            return (file_name, [])
        line_entropies = calculate_line_entropy(
            _global_model, _global_tokenizer, java_file_path,
            _global_device, max_seq_length, target_line_numbers,
            automaton=_global_automaton
        )
        return (file_name, line_entropies)
    except Exception as e:
        print(f"  Error: processing file {file_name} failed: {e}")
        return (file_name, [])


def process_version_directory(model, tokenizer, version_dir, version_name, output_dir, device='cuda',
                              num_workers=1, model_path=None, tokenizer_path=None,
                              max_seq_length=None, automaton: SyntaxAutomaton = None):
    if not HAS_TQDM:
        print(f"  Version directory: {version_dir}")

    n_gram_result_dir = '/root/workspace/lzc/SynDef/n_gram_result'
    n_gram_result_file = os.path.join(n_gram_result_dir, f"{version_name}-line-lvl-result.txt")
    n_gram_file_lines_map = load_n_gram_result_lines(n_gram_result_file)

    file_level_dir = '/root/workspace/lzc/SynDef/File-level'
    csv_file_path = os.path.join(file_level_dir, f"{version_name}_ground-truth-files_dataset.csv")

    if not os.path.exists(csv_file_path):
        print(f"  Warning: CSV file does not exist: {csv_file_path}, skipping this version")
        return 0

    buggy_file_paths = set()
    print(f"  Reading CSV file: {csv_file_path}")

    def is_test_file(file_path):
        if not file_path: return False
        file_path_lower = file_path.lower()
        if '/test/' in file_path_lower or '\\test\\' in file_path_lower: return True
        file_name = os.path.basename(file_path)
        if file_name.startswith('Test') or file_name.endswith('Test.java'): return True
        return False

    with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Bug', '').lower() == 'true':
                file_path = row.get('File', '').strip()
                if file_path and not is_test_file(file_path):
                    buggy_file_paths.add(file_path)

    print(f"  Found {len(buggy_file_paths)} Bug=true files (test files excluded)")

    java_files = []
    version_path = Path(version_dir)
    filename_to_csv_path = {}

    for csv_path in buggy_file_paths:
        expected_filename = csv_path.replace('/', '_').replace('\\', '_')
        java_file = version_path / expected_filename
        if java_file.exists() and java_file.is_file():
            java_files.append(java_file)
            filename_to_csv_path[java_file.name] = csv_path
        else:
            found_files = list(version_path.rglob(expected_filename))
            if found_files:
                java_files.append(found_files[0])
                filename_to_csv_path[found_files[0].name] = csv_path

    print(f"  Found {len(java_files)} matching Java files")
    if len(java_files) == 0:
        print(f"  Warning: no matching Java files found, skipping this version")
        return 0

    file_line_entropies = defaultdict(list)

    if num_workers > 1 and len(java_files) > 1:
        if model_path is None or tokenizer_path is None:
            print(f"  Warning: multi-process mode requires model_path and tokenizer_path, falling back to single-process")
            num_workers = 1

        if num_workers > 1:
            file_args = []
            for java_file in java_files:
                csv_path = filename_to_csv_path.get(java_file.name, None)
                target_lines = None
                if csv_path and csv_path in n_gram_file_lines_map:
                    target_lines = n_gram_file_lines_map[csv_path]
                elif java_file.name in n_gram_file_lines_map:
                    target_lines = n_gram_file_lines_map[java_file.name]

                if target_lines is not None:
                    file_args.append((str(java_file), java_file.name, max_seq_length, target_lines))
                elif not HAS_TQDM:
                    print(f"  Skipping file (not found in n_gram_result): {java_file.name}")

            with Pool(processes=num_workers, initializer=init_worker,
                     initargs=(model_path, tokenizer_path, device)) as pool:
                if HAS_TQDM:
                    results = list(tqdm(pool.imap(process_single_file, file_args),
                                       total=len(file_args), desc="  Processing files", unit="files", leave=False))
                else:
                    results = pool.map(process_single_file, file_args)

            for file_name, line_entropies in results:
                if line_entropies:
                    file_line_entropies[file_name] = line_entropies

    if num_workers == 1 or len(java_files) == 1:
        if HAS_TQDM:
            file_iterator = tqdm(java_files, desc=f"  Processing files", unit="files", leave=False)
        else:
            file_iterator = java_files

        for file_idx, java_file in enumerate(file_iterator, 1):
            file_name = java_file.name
            if not HAS_TQDM:
                print(f"  Processing file [{file_idx}/{len(java_files)}]: {file_name}")

            csv_path = filename_to_csv_path.get(file_name, None)
            target_lines = None
            if csv_path and csv_path in n_gram_file_lines_map:
                target_lines = n_gram_file_lines_map[csv_path]
                if not HAS_TQDM: print(f"    Need to compute {len(target_lines)} lines")
            elif file_name in n_gram_file_lines_map:
                target_lines = n_gram_file_lines_map[file_name]
                if not HAS_TQDM: print(f"    Need to compute {len(target_lines)} lines")
            else:
                if not HAS_TQDM: print(f"    Warning: file not found in n_gram_result, skipping")
                continue

            try:
                line_entropies = calculate_line_entropy(model, tokenizer, str(java_file), device,
                                                        max_seq_length, target_lines, automaton=automaton)
                if line_entropies:
                    file_line_entropies[file_name] = line_entropies
            except Exception as e:
                print(f"  Error: processing file {file_name} failed: {e}")
                import traceback; traceback.print_exc()
                continue

    print(f"  Merging and sorting results...")
    initial_results = []
    for file_name, line_entropies in file_line_entropies.items():
        original_path = filename_to_csv_path.get(file_name, file_name)
        normalized = []
        for item in line_entropies:
            if isinstance(item, tuple) and len(item) >= 2:
                normalized.append({
                    'line_num': item[0],
                    'line_entropy': float(item[1]),
                    'line_surprisal': 0.0,
                    'defect_element_count': 0,
                    'orig_entropy': 0.0,
                    'delta_entropy': 0.0,
                    'illegal_token_count': 0,
                    'token_count': 0
                })
            elif isinstance(item, dict):
                defect_count_val = item.get('defect_element_count')
                if defect_count_val is None and 'has_control' in item:
                    try: defect_count_val = int(item.get('has_control', 0))
                    except: defect_count_val = 0
                if defect_count_val is None and 'ast_depth' in item:
                    try: defect_count_val = 1 if int(item.get('ast_depth', 0)) > 0 else 0
                    except: defect_count_val = 0
                normalized.append({
                    'line_num': item.get('line_num'),
                    'line_entropy': float(item.get('line_entropy', 0.0)),
                    'line_surprisal': float(item.get('line_surprisal', 0.0)),
                    'defect_element_count': int(defect_count_val) if defect_count_val is not None else 0,
                    'orig_entropy': float(item.get('orig_entropy', 0.0)),
                    'delta_entropy': float(item.get('delta_entropy', 0.0)),
                    'illegal_token_count': int(item.get('illegal_token_count', 0)),
                    'token_count': int(item.get('token_count', 0))
                })
        scores = compute_optimized_scores(normalized)
        if not scores.size:
            scores = np.zeros(len(normalized), dtype=np.float32)
        order = np.argsort(-scores, kind='mergesort')
        compressed_scores = compress_optimized_scores(len(order))
        for new_idx, src_idx in enumerate(order):
            row = normalized[src_idx]
            row['optimized_score'] = compressed_scores[new_idx]
            predicted_buggy_line = f"{original_path}:{row['line_num']}"
            initial_results.append({
                'predicted_buggy_lines': predicted_buggy_line,
                'predicted_buggy_line_numbers': row['line_num'],
                'line_surprisal': row['line_surprisal'],
                'defect_element_count': row['defect_element_count'],
                'line_entropy': row['line_entropy'],
                'orig_entropy': row.get('orig_entropy', 0.0),
                'delta_entropy': row.get('delta_entropy', 0.0),
                'illegal_token_count': row.get('illegal_token_count', 0),
                'token_count': row.get('token_count', 0),
                'optimized_score': row['optimized_score']
            })

    print(f"  Applying automaton filtering rules...")
    automata_data = {'automata': []}
    automata_files = []
    for release in get_predefined_version_mapping().get(version_name, [version_name]):
        auto_path = Path('/root/workspace/lzc/automat/Line-auto_sourcedata') / release / f"{release}.json"
        if auto_path.exists():
            automata_files.append(str(auto_path))
    for path in automata_files:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            automata_data['automata'].extend(data.get('automata', []))
    automata_processor = UltraAutomataProcessor(automata_data)

    source_dir = os.path.join('/root/workspace/lzc/automat/sourcefile/sourcedata', version_name)
    file_cache = UltraFileCache(max_cache_size=400)
    lines_by_file = defaultdict(list)
    for row in initial_results:
        file_path, line_number_str = row['predicted_buggy_lines'].rsplit(':', 1)
        try: line_number = int(line_number_str)
        except ValueError: continue
        source_file_name = convert_csv_path_to_file_name(file_path)
        source_file_path = os.path.join(source_dir, source_file_name)
        line_content = file_cache.get_line(source_file_path, line_number)
        if line_content is None: continue
        tokens = tokenize_java_line_ultra_cached(line_content)
        can_reach_final = automata_processor.check_tokens_fast(tokens)
        lines_by_file[file_path].append((row, can_reach_final))

    results = []
    for file_path, entries in lines_by_file.items():
        min_opt, max_opt = None, None
        for row, _ in entries:
            opt = float(row.get('optimized_score', OPTIMIZED_SCORE_LOWER))
            if min_opt is None or opt < min_opt: min_opt = opt
            if max_opt is None or opt > max_opt: max_opt = opt
        processed_rows = []
        for row, can_reach_final in entries:
            new_row = dict(row)
            if can_reach_final:
                surprisal_val = safe_float(new_row.get('line_surprisal', 0.0))
                if surprisal_val != 0:
                    new_row['line_surprisal'] = 0.0
                base_score = safe_float(new_row.get('optimized_score', OPTIMIZED_SCORE_LOWER))
                illegal_count = int(safe_float(new_row.get('illegal_token_count', 0)))
                if min_opt is not None and max_opt is not None:
                    if illegal_count >= 3:
                        dst_min, dst_max = 0.0001, 0.4999
                    else:
                        dst_min, dst_max = 0.5001, 0.9999
                    scaled = scale_to_range(base_score, min_opt, max_opt, dst_min, dst_max)
                    if scaled is not None:
                        new_row['optimized_score'] = scaled
            processed_rows.append(new_row)

        for idx, row in enumerate(processed_rows):
            ls_val = safe_float(row.get('line_surprisal', 0.0))
            opt_val = safe_float(row.get('optimized_score', OPTIMIZED_SCORE_LOWER))
            token_val = safe_float(row.get('token_count', 0.0))
            defect_val = safe_float(row.get('defect_element_count', 0.0))
            row['_bucket'] = compute_rank_bucket(ls_val, opt_val)
            row['_token_key'] = -token_val
            row['_defect_key'] = -defect_val
            row['_original_idx'] = idx

        processed_rows.sort(key=lambda r: (r['_bucket'], r['_token_key'], r['_defect_key'], r['_original_idx']))
        for rank, row in enumerate(processed_rows, 1):
            row['rank'] = rank
            for k in ['_bucket', '_token_key', '_defect_key', '_original_idx']:
                row.pop(k, None)
            results.append(row)

    output_csv_path = os.path.join(output_dir, f"{version_name}-result.csv")
    if not HAS_TQDM: print(f"  Writing results to: {output_csv_path}")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['predicted_buggy_lines', 'predicted_buggy_line_numbers', 'rank', 'line_entropy',
                      'line_surprisal', 'defect_element_count', 'orig_entropy', 'delta_entropy',
                      'illegal_token_count', 'token_count', 'optimized_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if HAS_TQDM:
            results_iterator = tqdm(results, desc="  Writing CSV", unit="lines", leave=False)
        else:
            results_iterator = results
        for result in results_iterator:
            writer.writerow({
                'predicted_buggy_lines': result['predicted_buggy_lines'],
                'predicted_buggy_line_numbers': result['predicted_buggy_line_numbers'],
                'rank': result['rank'],
                'line_entropy': f"{result['line_entropy']:.10f}",
                'line_surprisal': f"{result['line_surprisal']:.10f}",
                'defect_element_count': result['defect_element_count'],
                'orig_entropy': f"{result.get('orig_entropy', 0.0):.10f}",
                'delta_entropy': f"{result.get('delta_entropy', 0.0):.10f}",
                'illegal_token_count': result.get('illegal_token_count', 0),
                'token_count': result.get('token_count', 0),
                'optimized_score': f"{result.get('optimized_score', OPTIMIZED_SCORE_LOWER):.10f}"
            })

    if not HAS_TQDM: print(f"  Completed version {version_name}! Processed {len(results)} lines")
    return len(results)


def main():
    model_dir = '/root/workspace/lzc/SynDef/transform-model/autoregressive_from_rawfiles'
    model_path = os.path.join(model_dir, 'best_model.pth')
    tokenizer_path = '/root/workspace/lzc/SynDef/transform-model/tokenizer_vocab.json'
    sourcedata_dir = '/root/workspace/lzc/SynDef/sourcedata'
    output_dir = '/root/workspace/lzc/SynDef/SynDef_result'

    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}"); return
    if not os.path.exists(tokenizer_path):
        print(f"Error: tokenizer file not found: {tokenizer_path}"); return
    if not os.path.exists(sourcedata_dir):
        print(f"Error: source data directory not found: {sourcedata_dir}"); return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_workers = int(os.environ.get('NUM_WORKERS', '1'))
    max_workers = cpu_count()
    if num_workers > 1:
        if device == 'cuda':
            print(f"⚠️  Warning: multi-process with GPU may cause GPU out-of-memory")
            force_multiprocess = os.environ.get('FORCE_MULTIPROCESS', '0') == '1'
            if not force_multiprocess:
                num_workers = 1
                print("Automatically switched to single-process mode (GPU mode)")
            else:
                print("⚠️  Forcing multi-process mode (may cause GPU out-of-memory)")
        else:
            if num_workers > max_workers:
                num_workers = max_workers
                print(f"⚠️  Workers exceed CPU cores ({max_workers}), adjusted to {num_workers}")
    if num_workers > 1:
        print(f"✅ Using {num_workers} processes in parallel (does not affect result accuracy)")
    else:
        print(f"Using single-process sequential processing")

    print("Loading tokenizer...")
    try:
        tokenizer = JavaCodeTokenizer()
        tokenizer.load_vocab(tokenizer_path)
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}"); import traceback; traceback.print_exc(); return

    print("Loading model...")
    try:
        from transform import TransformerModel, create_model
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        model_cfg = checkpoint.get('model_config', None)
        vocab_size = int(checkpoint.get('vocab_size', state_dict['embedding.weight'].shape[0]))
        if model_cfg:
            try:
                d_model = int(model_cfg['d_model'])
                n_heads = int(model_cfg['n_heads'])
                n_layers = int(model_cfg['n_layers'])
                d_ff = int(model_cfg['d_ff'])
                max_seq_length = int(model_cfg['max_seq_length'])
            except Exception:
                print(f"Error: model_config fields incomplete or incorrect type: {model_cfg}"); return
        else:
            from inspect import signature
            try:
                sig = signature(create_model)
                default_kwargs = {k: v.default for k, v in sig.parameters.items() if v.default is not sig.empty}
                default_n_heads = int(default_kwargs.get('n_heads', 8))
            except Exception:
                default_n_heads = 8
            d_model = int(state_dict['embedding.weight'].shape[1])
            n_layers = len([k for k in state_dict.keys() if 'transformer_blocks' in k and 'attention.W_q.weight' in k])
            d_ff = int(state_dict['transformer_blocks.0.feed_forward.0.weight'].shape[0]) if n_layers > 0 else int(default_kwargs.get('d_ff', 2048))
            max_seq_length = int(state_dict['positional_encoding.pe'].shape[1]) if 'positional_encoding.pe' in state_dict else int(default_kwargs.get('max_seq_length', 512))
            candidate_heads = [default_n_heads, 4, 8, 12, 16, 24, 32]
            n_heads = None
            for h in candidate_heads:
                if h > 0 and d_model % h == 0:
                    n_heads = h
                    break
            if n_heads is None:
                print(f"Error: cannot find suitable n_heads for d_model={d_model}. Please re-export weights with model_config included."); return
        if d_model <= 0 or n_heads <= 0 or n_layers <= 0 or d_ff <= 0 or max_seq_length <= 0:
            print(f"Error: invalid model configuration: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}, max_seq_length={max_seq_length}"); return
        if d_model % n_heads != 0:
            print(f"Error: model configuration inconsistent d_model={d_model}, n_heads={n_heads} (not divisible). Please fix the training config."); return

        print(f"Model configuration: vocab_size={vocab_size}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}, max_seq_length={max_seq_length}")
        model = TransformerModel(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_seq_length=max_seq_length, dropout=0.1, pad_token_id=tokenizer.pad_token_id
        )
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}"); import traceback; traceback.print_exc(); return

    print("Loading syntax automaton...")
    automaton = None
    try:
        automaton = SyntaxAutomaton(vocabulary_path=tokenizer_path)
        automaton.load('/root/workspace/lzc/SynDef/automat-model')
        print("Syntax automaton loaded successfully!")
    except Exception as e:
        print(f"Warning: syntax automaton loading failed, continuing without automaton filtering. Error: {e}")

    os.makedirs(output_dir, exist_ok=True)

    target_versions = [
        'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
        'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1',
        'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0',
        'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0',
        'lucene-3.1', 'wicket-1.5.3'
    ]

    print(f"\nStarting processing source data directory: {sourcedata_dir}")
    all_version_dirs = [d for d in os.listdir(sourcedata_dir) if os.path.isdir(os.path.join(sourcedata_dir, d))]
    version_dirs = [d for d in all_version_dirs if d in target_versions]
    version_dirs.sort()
    print(f"Found {len(all_version_dirs)} version directories, will process {len(version_dirs)} specified versions")
    if len(version_dirs) < len(target_versions):
        missing = set(target_versions) - set(version_dirs)
        if missing: print(f"Warning: following version directories missing: {sorted(missing)}")

    total_results = 0
    start_time = datetime.now()
    if HAS_TQDM:
        version_iterator = tqdm(enumerate(version_dirs, 1), total=len(version_dirs), desc="Processing versions", unit="versions")
    else:
        version_iterator = enumerate(version_dirs, 1)

    for version_idx, version_name in version_iterator:
        version_dir = os.path.join(sourcedata_dir, version_name)
        if not HAS_TQDM:
            print(f"\n[{version_idx}/{len(version_dirs)}] Processing version: {version_name}")
        try:
            num_results = process_version_directory(
                model=model, tokenizer=tokenizer, version_dir=version_dir, version_name=version_name,
                output_dir=output_dir, device=device, num_workers=num_workers,
                model_path=model_path if num_workers > 1 else None,
                tokenizer_path=tokenizer_path if num_workers > 1 else None,
                max_seq_length=max_seq_length, automaton=automaton
            )
            total_results += num_results
        except Exception as e:
            print(f"Error processing version {version_name}: {e}")
            import traceback; traceback.print_exc()
            continue

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n✅ All done!")
    print(f"  Processed {len(version_dirs)} versions")
    print(f"  Total {total_results} lines")
    print(f"  Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"  Results saved in: {output_dir}")

if __name__ == "__main__":
    main()