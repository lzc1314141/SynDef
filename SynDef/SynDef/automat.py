#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import random
import json
import pickle
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
from multiprocessing import Pool, cpu_count
from functools import partial
import csv
from pathlib import Path
import sys


try:
    import javalang
    from javalang.tree import *
    from javalang.parser import JavaSyntaxError
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("javalang not installed, using simplified syntax parsing. Recommended: pip install javalang")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


FILE_LEVEL_BASE_DIR = "/root/workspace/lzc/SynDef/File-level"



def parse_bool_str(value: str) -> bool:
    """
    Parse string to boolean.
    """
    if value is None:
        return False
    v = str(value).strip().lower()
    return v in {"true", "1", "yes", "y", "t"}


def sanitize_file_level_path(file_path: str) -> str:
    """
    Normalize file path and replace slashes with underscores.
    """
    if not file_path:
        return ""
    normalized = file_path.replace("\\", "/").lstrip("./")
    return normalized.replace("/", "_")


def find_file_level_csv_path(base_dir: str, version_key: str) -> Optional[Path]:
    """
    Find the CSV file for a given version in the file-level directory.
    """
    base = Path(base_dir)
    if not base.exists():
        logger.warning(f"File-level directory does not exist: {base_dir}")
        return None
    expected = base / f"{version_key}_ground-truth-files_dataset.csv"
    if expected.exists():
        return expected
    version_lower = version_key.lower()
    for candidate in base.glob("*.csv"):
        name = candidate.name.lower()
        if name.startswith(version_lower) and name.endswith("_ground-truth-files_dataset.csv"):
            return candidate
    logger.warning(f"CSV for version not found: {version_key} in {base_dir}")
    return None


def load_non_buggy_filenames_from_csv(csv_path: str) -> Set[str]:
    """
    Load non-buggy file names from CSV.
    """
    allowed: Set[str] = set()
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return allowed
    try:
        
        try:
            csv.field_size_limit(min(sys.maxsize, 1_000_000_000))
        except Exception:
            pass
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                logger.warning(f"CSV has no valid headers: {csv_path}")
                return allowed
            field_map = {name.lower(): name for name in reader.fieldnames}
            bug_col = field_map.get("bug")
            file_col = field_map.get("file")
            if not bug_col or not file_col:
                logger.warning(f"CSV missing required fields (File and Bug): {csv_path}")
                return allowed
            for row in reader:
                bug_val = row.get(bug_col, "")
                if parse_bool_str(bug_val):
                    continue
                file_path = row.get(file_col, "")
                if not file_path or not file_path.strip():
                    continue
                if not file_path.endswith(".java"):
                    continue
                sanitized = sanitize_file_level_path(file_path)
                if sanitized:
                    allowed.add(sanitized)
    except Exception as e:
        logger.warning(f"Failed to read CSV: {csv_path} | Error: {e}")
    return allowed


class ParseState:
    """
    Represents a parsing state with a context stack.
    """
    def __init__(self, context_stack: List[str] = None):
        
        self.context_stack = context_stack or []
    
    def push(self, context: str):
        
        self.context_stack.append(context)
    
    def pop(self):
        
        if self.context_stack:
            self.context_stack.pop()
    
    def get_state_id(self) -> str:
        
        return '|'.join(self.context_stack) if self.context_stack else 'root'
    
    def copy(self):
        
        return ParseState(self.context_stack.copy())
    
    def __hash__(self):
        return hash(self.get_state_id())
    
    def __eq__(self, other):
        return isinstance(other, ParseState) and self.context_stack == other.context_stack
    
    def __repr__(self):
        return f"ParseState({self.context_stack})"


class SyntaxAutomaton:
    """
    Syntax automaton that learns legal token sequences from Java files.
    """
    
    def __init__(self, vocabulary_path: str = None):
        
        self.vocabulary = {}
        self.id_to_token = {}
        self.vocab_size = 0
        
        # State machine: state -> {token_id: count}
        self.state_transitions = defaultdict(lambda: defaultdict(int))
        
        # State statistics: state -> total_count
        self.state_counts = defaultdict(int)
        
        # Map from state to legal tokens (computed after training)
        self.state_legal_tokens = {}
        
        # Training history records
        self.training_history = {
            'files_processed': [],
            'states_count': [],
            'transitions_count': [],
            'vocab_size': [],
            'legal_tokens_count': []
        }
        
        # Mapping of state type names to descriptions
        self.state_types = {
            'class': 'class declaration',
            'interface': 'interface declaration',
            'enum': 'enum declaration',
            'annotation': 'annotation declaration',
            'method': 'method declaration',
            'throws': 'throws clause',
            'type_params_decl': 'type parameters declaration',
            'type_params_use': 'generic type usage',
            'annotation_args': 'annotation arguments',
            'module': 'module declaration',
            'module_directive': 'module directive',
            'text_block': 'text block',
            'record': 'record declaration',
            'sealed': 'sealed class declaration',
            'permits': 'permits clause',
            'if': 'if condition',
            'for': 'for loop',
            'foreach': 'enhanced for loop',
            'do_while': 'do-while loop',
            'switch': 'switch statement',
            'while': 'while loop',
            'try': 'try block',
            'try_with_resources': 'try-with-resources block',
            'synchronized': 'synchronized block',
            'lambda': 'lambda expression',
            'throw': 'throw statement',
            'catch': 'catch block',
            'expression': 'expression',
            'statement': 'statement',
            'field': 'field declaration',
            'constructor': 'constructor',
            'package': 'package declaration',
            'import': 'import declaration',
            'root': 'root node'
        }
        
        if vocabulary_path and os.path.exists(vocabulary_path):
            self.load_vocabulary(vocabulary_path)
    
    def load_vocabulary(self, vocabulary_path: str):
        
        logger.info(f"Loading vocabulary: {vocabulary_path}")
        with open(vocabulary_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.vocabulary = vocab_data.get('token_to_id', {})
            self.id_to_token = {v: k for k, v in self.vocabulary.items()}
            self.vocab_size = len(self.vocabulary)
        logger.info(f"Vocabulary size: {self.vocab_size}")
    
    def build_vocabulary(self, train_data_dir: str, min_count: int = 1):
        
        if self.vocabulary:
            logger.info(f"Vocabulary already exists (size: {len(self.vocabulary)}), skipping build")
            return
        
        logger.info("Building vocabulary...")
        token_counts = Counter()
        
        # Collect all Java files
        java_files = []
        for root, dirs, files in os.walk(train_data_dir):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        
        logger.info(f"Scanning {len(java_files)} files to build vocabulary...")
        
        # Count tokens across all files
        with tqdm(total=len(java_files), desc="Building vocabulary", unit="files", ncols=100) as pbar:
            for file_path in java_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    tokens = self.tokenize_java_code(code)
                    for token, _, _ in tokens:
                        token_counts[token] += 1
                except Exception:
                    pass
                pbar.update(1)
        
        # Build vocabulary: special tokens + frequent tokens
        self.vocabulary = {
            '[PAD]': 0,
            '[MASK]': 1,
            '[UNK]': 2,
            '[CLS]': 3,
            '[SEP]': 4
        }
        
        
        next_id = 5
        for token, count in token_counts.most_common():
            if count >= min_count and token not in self.vocabulary:
                self.vocabulary[token] = next_id
                next_id += 1
        
        self.id_to_token = {v: k for k, v in self.vocabulary.items()}
        self.vocab_size = len(self.vocabulary)
        
        logger.info(f"Vocabulary built, size: {self.vocab_size}")
        logger.info(f"  Special tokens: 5 ([PAD], [MASK], [UNK], [CLS], [SEP])")
        logger.info(f"  Regular tokens: {self.vocab_size - 5} (count >= {min_count})")
    
    def build_vocabulary_from_files(self, java_files: List[str], min_count: int = 1):
        
        if self.vocabulary:
            logger.info(f"Vocabulary already exists (size: {len(self.vocabulary)}), skipping build")
            return
        logger.info("Building vocabulary from filtered file list...")
        token_counts = Counter()
        with tqdm(total=len(java_files), desc="Building vocabulary", unit="files", ncols=100) as pbar:
            for file_path in java_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    tokens = self.tokenize_java_code(code)
                    for token, _, _ in tokens:
                        token_counts[token] += 1
                except Exception:
                    pass
                pbar.update(1)
        self.vocabulary = {
            '[PAD]': 0,
            '[MASK]': 1,
            '[UNK]': 2,
            '[CLS]': 3,
            '[SEP]': 4
        }
        next_id = 5
        for token, count in token_counts.most_common():
            if count >= min_count and token not in self.vocabulary:
                self.vocabulary[token] = next_id
                next_id += 1
        self.id_to_token = {v: k for k, v in self.vocabulary.items()}
        self.vocab_size = len(self.vocabulary)
        logger.info(f"Vocabulary built, size: {self.vocab_size} (from non-buggy files)")
        logger.info(f"  Special tokens: 5 ([PAD], [MASK], [UNK], [CLS], [SEP])")
        logger.info(f"  Regular tokens: {self.vocab_size - 5} (count >= {min_count})")
    
    def tokenize_java_code(self, code: str) -> List[Tuple[str, int, int]]:
        
        import re
        from bisect import bisect_right
        tokens: List[Tuple[str, int, int]] = []
        
        
        def _handle_string_literals(src: str) -> str:
            
            src = re.sub(r'"""[\s\S]*?"""', '<TEXT_BLOCK>', src)
            src = re.sub(r'"(?:[^"\\]|\\.)*"', '<STRING>', src)
            src = re.sub(r"'(?:[^'\\]|\\.)*'", '<CHAR>', src)
            return src
        
        def _handle_comments(src: str) -> str:
            src = re.sub(r'//.*?$', '', src, flags=re.MULTILINE)
            src = re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)
            return src
        
        processed_code = _handle_string_literals(code)
        processed_code = _handle_comments(processed_code)
        
        
        pattern = r'''
            (?://.*?$)|                          
            (?:/\*.*?\*/)|                       
            (?:"(?:[^"\\]|\\.)*")|               
            (?:'(?:[^'\\]|\\.)*')|               
            (?:\d+\.?\d*[fFdD]?)|                
            (?:0[xX][0-9a-fA-F]+)|               
            (?:\w+)|                             
            (?:[{};(),\[\]<>])|                 
            (?:[+\-*/=!<>&|^%])|                 
            (?:\.)|                              
            (?:\s+)                              
        '''
        
        
        line_starts = [0]
        for i, ch in enumerate(processed_code):
            if ch == '\n':
                line_starts.append(i + 1)
        
        if not line_starts:
            line_starts = [0]
        
        for match in re.finditer(pattern, processed_code, re.MULTILINE | re.VERBOSE):
            raw = match.group()
            token = raw.strip()
            if not token or token.isspace():
                continue
            start_idx = match.start()
            
            line_idx = bisect_right(line_starts, start_idx) - 1
            line_num = line_idx + 1
            col_num = start_idx - line_starts[line_idx]
            tokens.append((token, line_num, col_num))
        
        return tokens
    
    def parse_java_code(self, code: str):
        
        if not JAVALANG_AVAILABLE:
            return None
        try:
            tree = javalang.parse.parse(code)
            return tree
        except Exception as e:
            logger.debug(f"Java parsing error: {e}")
            return None
    
    def extract_parse_states(self, tree, tokens: List[Tuple[str, int, int]]) -> List[Tuple[int, ParseState]]:
       
        
        token_states = {}
        
        def get_node_context(node):
            
            if not JAVALANG_AVAILABLE:
                return None
            try:
                from javalang.tree import (
                    ClassDeclaration, MethodDeclaration, ConstructorDeclaration,
                    FieldDeclaration, IfStatement, ForStatement, WhileStatement,
                    TryStatement, CatchClause, Expression, BinaryOperation,
                    MemberReference, MethodInvocation, Statement,
                    LocalVariableDeclaration, ReturnStatement, PackageDeclaration, Import,
                    InterfaceDeclaration, EnumDeclaration, AnnotationDeclaration,
                    SwitchStatement, DoStatement, LambdaExpression, SynchronizedStatement,
                    ThrowStatement
                )
                if isinstance(node, ClassDeclaration):
                    return 'class'
                elif isinstance(node, InterfaceDeclaration):
                    return 'interface'
                elif isinstance(node, EnumDeclaration):
                    return 'enum'
                elif isinstance(node, AnnotationDeclaration):
                    return 'annotation'
                elif isinstance(node, MethodDeclaration):
                    return 'method'
                elif isinstance(node, ConstructorDeclaration):
                    return 'constructor'
                elif isinstance(node, FieldDeclaration):
                    return 'field'
                elif isinstance(node, IfStatement):
                    return 'if'
                elif isinstance(node, ForStatement):
                    
                    try:
                        control = getattr(node, 'control', None)
                        if control and control.__class__.__name__ == 'EnhancedForControl':
                            return 'foreach'
                    except Exception:
                        pass
                    return 'for'
                elif isinstance(node, WhileStatement):
                    return 'while'
                elif isinstance(node, DoStatement):
                    return 'do_while'
                elif isinstance(node, SwitchStatement):
                    return 'switch'
                elif isinstance(node, TryStatement):
                    # try-with-resources
                    try:
                        resources = getattr(node, 'resources', None)
                        if resources:
                            return 'try_with_resources'
                    except Exception:
                        pass
                    return 'try'
                elif isinstance(node, CatchClause):
                    return 'catch'
                elif isinstance(node, SynchronizedStatement):
                    return 'synchronized'
                elif isinstance(node, LambdaExpression):
                    return 'lambda'
                elif isinstance(node, ThrowStatement):
                    return 'throw'
                elif isinstance(node, (Expression, BinaryOperation, MemberReference, MethodInvocation)):
                    return 'expression'
                elif isinstance(node, (Statement, LocalVariableDeclaration, ReturnStatement)):
                    return 'statement'
                elif isinstance(node, PackageDeclaration):
                    return 'package'
                elif isinstance(node, Import):
                    return 'import'
            except:
                pass
            return None
        
        def get_node_line_span(node) -> Optional[Tuple[int, int]]:
            
            min_line: Optional[int] = None
            max_line: Optional[int] = None
            
            def update_from(n):
                nonlocal min_line, max_line
                pos = getattr(n, 'position', None)
                if pos and isinstance(pos, tuple) and len(pos) >= 1 and isinstance(pos[0], int):
                    line = pos[0]
                    if min_line is None or line < min_line:
                        min_line = line
                    if max_line is None or line > max_line:
                        max_line = line
            
            def walk(n):
                update_from(n)
                # traverse children
                if hasattr(n, 'children'):
                    for child in n.children:
                        if child is None:
                            continue
                        if isinstance(child, list):
                            for item in child:
                                if item is not None:
                                    walk(item)
                        else:
                            walk(child)
            
            try:
                walk(node)
            except Exception:
                
                update_from(node)
            
            if min_line is None and max_line is None:
                return None
            if min_line is None:
                min_line = max_line
            if max_line is None:
                max_line = min_line
            if min_line > max_line:
                min_line, max_line = max_line, min_line
            return (min_line, max_line)
        
        def traverse_node(node, state_stack: List[str]):
            
            context = get_node_context(node)
            
            try:
                cls_name = node.__class__.__name__ if node is not None else ''
                
                if not context:
                    if cls_name in ('ModuleDeclaration', 'Module'):
                        context = 'module'
                    elif cls_name in ('RecordDeclaration', 'Record'):
                        context = 'record'
                    elif cls_name in ('SealedClassDeclaration', 'SealedInterfaceDeclaration', 'Sealed'):
                        context = 'sealed'
            except Exception:
                pass
            
            if context == 'method':
                try:
                    throws_list = getattr(node, 'throws', None)
                    if throws_list:
                        
                        method_stack = state_stack + ['method', 'throws']
                    else:
                        method_stack = state_stack + ['method']
                except Exception:
                    method_stack = state_stack + ['method']
                state_stack = method_stack
            if context:
                state_stack = state_stack + [context]
            
            
            node_span = get_node_line_span(node)
            if node_span:
                start_line, end_line = node_span
                
                for i, (_, token_line, _) in enumerate(tokens):
                    if start_line <= token_line <= end_line:
                        prev = token_states.get(i)
                        if (prev is None) or (len(state_stack) > len(prev)):
                            token_states[i] = state_stack.copy()
            
            
            if hasattr(node, 'children'):
                for child in node.children:
                    if child is not None:
                        if isinstance(child, list):
                            for item in child:
                                if item is not None:
                                    traverse_node(item, state_stack)
                        else:
                            traverse_node(child, state_stack)
        
        
        if tree:
            try:
                traverse_node(tree, [])
            except Exception as e:
                logger.debug(f"AST traversal error: {e}")
        
        
        try:
            tokens_only = [t for (t, _, _) in tokens]
            
            def prev_significant_index(idx: int) -> int:
                j = idx - 1
                while j >= 0:
                    tok = tokens_only[j]
                    
                    if tok:
                        return j
                    j -= 1
                return -1
            
            def find_matching(idx_start: int, open_tok: str, close_tok: str) -> int:
                
                depth = 0
                for k in range(idx_start, len(tokens_only)):
                    if tokens_only[k] == open_tok:
                        depth += 1
                    elif tokens_only[k] == close_tok:
                        depth -= 1
                        if depth == 0:
                            return k
                return -1
            
            def mark_range(start_idx: int, end_idx: int, label: str):
                if start_idx < 0 or end_idx < 0:
                    return
                for i_mark in range(start_idx, end_idx + 1):
                    prev = token_states.get(i_mark, [])
                    
                    if isinstance(prev, list):
                        token_states[i_mark] = prev + [label]
                    else:
                        token_states[i_mark] = [label]
            
            
            for i, tok in enumerate(tokens_only):
                if tok == '<TEXT_BLOCK>':
                    mark_range(i, i, 'text_block')
            
            
            i = 0
            while i < len(tokens_only):
                tok = tokens_only[i]
                if tok == '@':
                    
                    j = i + 1
                    
                    while j < len(tokens_only) and tokens_only[j] not in ('(', ')', '{', '}', ';'):
                        j += 1
                    if j < len(tokens_only) and tokens_only[j] == '(':
                        close_j = find_matching(j, '(', ')')
                        if close_j != -1:
                            mark_range(j, close_j, 'annotation_args')
                            i = close_j
                i += 1
            
           
            i = 0
            while i < len(tokens_only):
                tok = tokens_only[i]
                if tok == 'module':
                   
                    brace_idx = -1
                    for k in range(i, len(tokens_only)):
                        if tokens_only[k] == '{':
                            brace_idx = k
                            break
                    if brace_idx != -1:
                        end_brace = find_matching(brace_idx, '{', '}')
                        if end_brace != -1:
                            mark_range(i, end_brace, 'module')
                           
                            directives = {'requires', 'exports', 'opens', 'uses', 'provides'}
                            p = brace_idx + 1
                            while p < end_brace:
                                if tokens_only[p] in directives:
                                    q = p
                                    while q <= end_brace and tokens_only[q] != ';':
                                        q += 1
                                    if q <= end_brace:
                                        mark_range(p, q, 'module_directive')
                                        p = q
                                p += 1
                            i = end_brace
                i += 1
            
            
            i = 0
            while i < len(tokens_only):
                tok = tokens_only[i]
                if tok == 'record':
                   
                    brace_idx = -1
                    for k in range(i, len(tokens_only)):
                        if tokens_only[k] == '{':
                            brace_idx = k
                            break
                    if brace_idx != -1:
                        end_brace = find_matching(brace_idx, '{', '}')
                        if end_brace != -1:
                            mark_range(i, end_brace, 'record')
                            i = end_brace
                i += 1
            
            
            i = 0
            while i < len(tokens_only):
                tok = tokens_only[i]
                if tok in ('sealed', 'non-sealed'):
                    
                    brace_idx = -1
                    for k in range(i, len(tokens_only)):
                        if tokens_only[k] == '{':
                            brace_idx = k
                            break
                    if brace_idx != -1:
                        end_brace = find_matching(brace_idx, '{', '}')
                        if end_brace != -1:
                            mark_range(i, end_brace, 'sealed')
                            i = end_brace
                if tok == 'permits':
                   
                    end_idx = -1
                    for k in range(i, len(tokens_only)):
                        if tokens_only[k] in ('{', ';'):
                            end_idx = k
                            break
                    if end_idx == -1:
                        end_idx = i
                    mark_range(i, end_idx, 'permits')
                i += 1
            
           
            decl_keywords = {'class', 'interface', 'enum', 'record'}
            i = 0
            while i < len(tokens_only):
                if tokens_only[i] == '<':
                    j = find_matching(i, '<', '>')
                    if j != -1:
                        # look back to decide context
                        back = max(0, i - 10)
                        window = set(tokens_only[back:i])
                        if window & decl_keywords:
                            mark_range(i, j, 'type_params_decl')
                        else:
                            mark_range(i, j, 'type_params_use')
                        i = j
                i += 1
        except Exception as _e:
            
            pass
        
        
        state_positions = []
        for i, (token, start_pos, end_pos) in enumerate(tokens):
            if i in token_states:
                state = ParseState(token_states[i])
            else:
                
                state = ParseState(['root'])
            state_positions.append((i, state))
        
        return state_positions
    
    @staticmethod
    def _process_file_worker(args):
        
        file_path, vocabulary = args
        state_transitions = defaultdict(lambda: defaultdict(int))
        state_counts = defaultdict(int)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            
            temp_automaton = SyntaxAutomaton()
            temp_automaton.vocabulary = vocabulary
            
            
            tokens = temp_automaton.tokenize_java_code(code)
            if not tokens:
                return (dict(state_transitions), dict(state_counts))
            
            
            tree = temp_automaton.parse_java_code(code)
            
            
            state_positions = temp_automaton.extract_parse_states(tree, tokens)
            
            
            for i in range(len(tokens) - 1):
                token, _, _ = tokens[i]
                next_token, _, _ = tokens[i + 1]
                
                
                if i < len(state_positions):
                    _, state = state_positions[i]
                else:
                    state = ParseState(['root'])
                
                state_id = state.get_state_id()
                
               
                unk_id = vocabulary.get('[UNK]', 2)
                token_id = vocabulary.get(token, unk_id)
                next_token_id = vocabulary.get(next_token, unk_id)
                
                
                state_transitions[state_id][token_id] += 1
                state_counts[state_id] += 1
                
                
                if i + 1 < len(state_positions):
                    _, next_state = state_positions[i + 1]
                    next_state_id = next_state.get_state_id()
                    state_transitions[state_id][next_token_id] += 1
                    
        except Exception as e:
            
            pass
        
        
        state_transitions_dict = {}
        for state_id, transitions in state_transitions.items():
            state_transitions_dict[state_id] = dict(transitions)
        return (state_transitions_dict, dict(state_counts))
    
    def learn_from_file(self, file_path: str):
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            
            tokens = self.tokenize_java_code(code)
            if not tokens:
                return
            
            
            tree = self.parse_java_code(code)
            
            
            state_positions = self.extract_parse_states(tree, tokens)
            
            
            for i in range(len(tokens) - 1):
                token, _, _ = tokens[i]
                next_token, _, _ = tokens[i + 1]
                
                
                if i < len(state_positions):
                    _, state = state_positions[i]
                else:
                    state = ParseState(['root'])
                
                state_id = state.get_state_id()
                
               
                unk_id = self.vocabulary.get('[UNK]', 2)
                token_id = self.vocabulary.get(token, unk_id)
                next_token_id = self.vocabulary.get(next_token, unk_id)
                
                
                self.state_transitions[state_id][token_id] += 1
                self.state_counts[state_id] += 1
                
                
                if i + 1 < len(state_positions):
                    _, next_state = state_positions[i + 1]
                    next_state_id = next_state.get_state_id()
                    
                    self.state_transitions[state_id][next_token_id] += 1
                
        except Exception as e:
            logger.debug(f"Error learning from file {file_path}: {e}")
    
    def compute_legal_token_masks(self, min_count: int = 1):
        
        logger.info("Computing legal token masks...")
        self.state_legal_tokens = {}
        
        with tqdm(self.state_transitions.items(), desc="Computing legal token masks", unit="states", ncols=100) as pbar:
            for state_id, transitions in pbar:
                legal_tokens = set()
                total_count = self.state_counts[state_id]
                
                for token_id, count in transitions.items():
                    
                    if count >= min_count or (total_count > 0 and count / total_count > 0.001):
                        legal_tokens.add(token_id)
                
                self.state_legal_tokens[state_id] = legal_tokens
                pbar.set_postfix({'current_state': state_id[:30], 'legal_tokens': len(legal_tokens)})
        
        logger.info(f"Computation complete, {len(self.state_legal_tokens)} states")
    
    def predict_legal_tokens(self, parse_state: ParseState) -> np.ndarray:
        
        state_id = parse_state.get_state_id()
        legal_tokens = self.state_legal_tokens.get(state_id, set())
        
        
        mask = np.zeros(self.vocab_size, dtype=np.float32)
        for token_id in legal_tokens:
            if 0 <= token_id < self.vocab_size:
                mask[token_id] = 1.0
        
        
        if not legal_tokens:
            
            common_tokens = [0, 1, 2, 3, 4]  # PAD, MASK, UNK, CLS, SEP
            for tid in common_tokens:
                if 0 <= tid < self.vocab_size:
                    mask[tid] = 1.0
        
        return mask
    
    def get_parse_state(self, parse_state: ParseState) -> Dict:
       
        state_id = parse_state.get_state_id()
        contexts = parse_state.context_stack
        
        
        translated_contexts = []
        for ctx in contexts:
            translated_contexts.append(self.state_types.get(ctx, ctx))
        
        return {
            'state_id': state_id,
            'context_stack': contexts,
            'translated_contexts': translated_contexts,
            'depth': len(contexts),
            'legal_token_count': len(self.state_legal_tokens.get(state_id, set()))
        }
    
    def train(self, train_data_dir: str, epochs: int = 20, min_count: int = 1, log_interval: int = 100, 
              num_workers: int = 8, early_stopping_patience: int = 3, convergence_threshold: float = 0.01,
              val_split: float = 0.1, max_val_files: int = 300, acc_plateau_delta: float = 0.001,
              stability_threshold: float = 0.98):
        
        logger.info(f"Starting training, data directory: {train_data_dir}")
        logger.info(f"Epochs: {epochs} (maximum)")
        logger.info(f"Using {num_workers} processes for parallel processing")
        logger.info(f"Early stopping patience: {early_stopping_patience} epochs, convergence threshold: {convergence_threshold*100:.1f}%")
        logger.info(f"Validation split: {val_split*100:.1f}%, max validation files: {max_val_files}")
        
        # Reset training history (per epoch)
        self.training_history = {
            'epoch': [],
            'states_count': [],
            'transitions_count': [],
            'vocab_size': [],
            'legal_tokens_count': [],
            'total_files_processed': [],
            # Quality and stability metrics
            'val_accuracy': [],
            'state_stability': [],
            'coverage_new_ratio': []
        }
        
        
        logger.info("Collecting non-buggy Java files...")
        data_path = Path(train_data_dir)
        if not data_path.exists():
            logger.error(f"Training data directory does not exist: {train_data_dir}")
            return
        java_files: List[str] = []
        non_buggy_cache: Dict[str, Set[str]] = {}
        missing_versions: Set[str] = set()
        skipped_buggy = 0
        skipped_missing = 0
        for java_file in data_path.rglob("*.java"):
            rel_parts = java_file.relative_to(data_path).parts
            if not rel_parts:
                continue
            version_key = rel_parts[0]
            file_name = java_file.name
            
            if version_key in missing_versions:
                skipped_missing += 1
                continue
            if version_key not in non_buggy_cache:
                csv_path = find_file_level_csv_path(FILE_LEVEL_BASE_DIR, version_key)
                if csv_path is None:
                    missing_versions.add(version_key)
                    skipped_missing += 1
                    continue
                non_buggy_cache[version_key] = load_non_buggy_filenames_from_csv(str(csv_path))
            allowed_files = non_buggy_cache.get(version_key, set())
            if not allowed_files:
                skipped_missing += 1
                continue
            if file_name not in allowed_files:
                skipped_buggy += 1
                continue
            java_files.append(str(java_file))
        logger.info(f"Found non-buggy Java files: {len(java_files)} (skipped buggy/unknown: {skipped_buggy}, missing CSV: {skipped_missing})")
        
        if len(java_files) < 100:
            logger.warning(f"⚠️  Training files are few ({len(java_files)}), may not fully learn Java syntax patterns")
            logger.warning("   It is recommended to use at least 100+ Java files for training")
        
        
        if not self.vocabulary:
            logger.info("\nVocabulary is empty, building vocabulary from non-buggy files only...")
            self.build_vocabulary_from_files(java_files, min_count)
            logger.info("")
        else:
            logger.info(f"Using existing vocabulary, size: {self.vocab_size}")
        
        
        rng = random.Random(42)
        shuffled = list(java_files)
        rng.shuffle(shuffled)
        val_size = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 10 else max(1, len(shuffled)//10 or 1)
        val_files = shuffled[:val_size]
        train_files = shuffled[val_size:]
        if len(val_files) > max_val_files:
            val_files = val_files[:max_val_files]
        logger.info(f"Train/validation split: train {len(train_files)} files, validation {len(val_files)} files")
        
        
        no_improvement_count = 0
        prev_states = 0
        prev_transitions = 0
        prev_val_acc = None
        prev_legal_snapshot = None
        
        def _compute_temp_legal_tokens() -> Dict[str, Set[int]]:
            temp_legal: Dict[str, Set[int]] = {}
            for state_id, transitions in self.state_transitions.items():
                total_count = self.state_counts[state_id]
                legal_tokens = set()
                for token_id, count in transitions.items():
                    if count >= min_count or (total_count > 0 and count / total_count > 0.001):
                        legal_tokens.add(token_id)
                temp_legal[state_id] = legal_tokens
            return temp_legal
        
        def _eval_on_files(files: List[str], temp_legal: Dict[str, Set[int]]) -> float:
            total = 0
            correct = 0
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    tokens = self.tokenize_java_code(code)
                    if not tokens:
                        continue
                    
                    tree = self.parse_java_code(code)
                    state_positions = self.extract_parse_states(tree, tokens)
                    for idx, parse_state in state_positions:
                        token_str = tokens[idx][0]
                        token_id = self.vocabulary.get(token_str, self.vocabulary.get('[UNK]', 2))
                        state_id = parse_state.get_state_id()
                        legal = temp_legal.get(state_id, set())
                        total += 1
                        if token_id in legal:
                            correct += 1
                except Exception:
                    continue
            return (correct / total) if total > 0 else 0.0
        
        def _compute_stability(prev_map: Dict[str, Set[int]], cur_map: Dict[str, Set[int]]) -> Tuple[float, float]:
            if not prev_map:
                return (0.0, 1.0)
            states = set(prev_map.keys()) | set(cur_map.keys())
            if not states:
                return (1.0, 0.0)
            jaccs = []
            changed = 0
            for s in states:
                a = prev_map.get(s, set())
                b = cur_map.get(s, set())
                if not a and not b:
                    jacc = 1.0
                else:
                    inter = len(a & b)
                    union = len(a | b)
                    jacc = (inter / union) if union > 0 else 1.0
                jaccs.append(jacc)
                if jacc < 0.9:
                    changed += 1
            avg_jacc = sum(jaccs) / len(jaccs)
            change_ratio = changed / len(states)
            return (avg_jacc, change_ratio)
        
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info(f"{'='*60}")
            
            if num_workers > 1 and len(train_files) > 100:
                
                logger.info(f"Using multi-process parallel processing ({num_workers} processes)")
                
                
                file_args = [(file_path, self.vocabulary) for file_path in train_files]
                
                
                with Pool(processes=num_workers) as pool:
                    
                    with tqdm(total=len(train_files), desc=f"Epoch {epoch}/{epochs}", unit="files", ncols=100) as pbar:
                        for i, (state_transitions_dict, state_counts_dict) in enumerate(
                            pool.imap_unordered(self._process_file_worker, file_args)
                        ):
                            
                            for state_id, transitions in state_transitions_dict.items():
                                for token_id, count in transitions.items():
                                    self.state_transitions[state_id][token_id] += count
                            for state_id, count in state_counts_dict.items():
                                self.state_counts[state_id] += count
                            
                            
                            pbar.update(1)
                            pbar.set_postfix({
                                'states': len(self.state_transitions),
                                'transitions': sum(len(transitions) for transitions in self.state_transitions.values())
                            })
            else:
                
                logger.info("Using single-process processing")
                with tqdm(total=len(train_files), desc=f"Epoch {epoch}/{epochs}", unit="files", ncols=100) as pbar:
                    
                    for i, file_path in enumerate(train_files):
                        self.learn_from_file(file_path)
                        
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'states': len(self.state_transitions),
                            'transitions': sum(len(transitions) for transitions in self.state_transitions.values())
                        })
            
            
            num_states = len(self.state_transitions)
            num_transitions = sum(len(transitions) for transitions in self.state_transitions.values())
            vocab_size = len(self.vocabulary)
            total_files = epoch * len(train_files)
            
            self.training_history['epoch'].append(epoch)
            self.training_history['states_count'].append(num_states)
            self.training_history['transitions_count'].append(num_transitions)
            self.training_history['vocab_size'].append(vocab_size)
            self.training_history['total_files_processed'].append(total_files)
            
            
            temp_legal = _compute_temp_legal_tokens()
            val_acc = _eval_on_files(val_files, temp_legal)
            avg_jacc, change_ratio = _compute_stability(prev_legal_snapshot, temp_legal)
            states_growth = (num_states - prev_states) / max(prev_states, 1) if prev_states > 0 else 1.0
            transitions_growth = (num_transitions - prev_transitions) / max(prev_transitions, 1) if prev_transitions > 0 else 1.0
            coverage_new_ratio = (max(num_states - prev_states, 0) + max(num_transitions - prev_transitions, 0)) / max(prev_states + prev_transitions, 1)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['state_stability'].append(avg_jacc)
            self.training_history['coverage_new_ratio'].append(coverage_new_ratio)
            
            logger.info(f"Epoch {epoch} complete - states: {num_states} (growth: {states_growth*100:+.2f}%), "
                       f"transitions: {num_transitions} (growth: {transitions_growth*100:+.2f}%), "
                       f"validation accuracy: {val_acc*100:.2f}%, state stability: {avg_jacc*100:.2f}%, "
                       f"new-pattern ratio: {coverage_new_ratio*100:.2f}%, vocab size: {vocab_size}")
            
            
            if epoch > 1:
                max_growth = max(states_growth, transitions_growth)
                acc_plateau = (prev_val_acc is not None) and ((val_acc - prev_val_acc) < acc_plateau_delta)
                stability_ok = (avg_jacc >= stability_threshold) or (change_ratio < 0.01)
                if max_growth < convergence_threshold and acc_plateau and stability_ok:
                    no_improvement_count += 1
                    logger.info(f"⚠️  Composite convergence conditions met (growth {max_growth*100:.2f}% < {convergence_threshold*100:.1f}%, "
                               f"validation improvement < {acc_plateau_delta*100:.2f}%, stability ≥ {stability_threshold*100:.1f}%), "
                               f"consecutive {no_improvement_count}/{early_stopping_patience} epochs")
                    if no_improvement_count >= early_stopping_patience:
                        logger.info(f"✓ Model converged (quantity + quality + stability), early stopping at epoch {epoch}")
                        logger.info("  States and transitions have stabilized, further training yields limited benefit")
                        break
                else:
                    no_improvement_count = 0  
            
            prev_states = num_states
            prev_transitions = num_transitions
            prev_val_acc = val_acc
            prev_legal_snapshot = temp_legal
        
        
        actual_epochs = len(self.training_history['epoch'])
        logger.info("\nComputing legal token masks...")
        self.compute_legal_token_masks(min_count)
        
        
        total_legal = sum(len(tokens) for tokens in self.state_legal_tokens.values())
        self.training_history['legal_tokens_count'] = [total_legal] * actual_epochs
        
        logger.info(f"\nTraining complete! Actual epochs: {actual_epochs} (max {epochs})")
        logger.info(f"Final states: {len(self.state_transitions)}")
        logger.info(f"Final transitions: {sum(len(transitions) for transitions in self.state_transitions.values())}")
        logger.info(f"Final total legal tokens: {total_legal}")
    
    def save(self, model_dir: str):
        
        os.makedirs(model_dir, exist_ok=True)
        
        
        state_transitions_path = os.path.join(model_dir, 'state_transitions.pkl')
        with open(state_transitions_path, 'wb') as f:
            pickle.dump(dict(self.state_transitions), f)
        
        
        state_counts_path = os.path.join(model_dir, 'state_counts.pkl')
        with open(state_counts_path, 'wb') as f:
            pickle.dump(dict(self.state_counts), f)
        
        
        legal_tokens_path = os.path.join(model_dir, 'state_legal_tokens.pkl')
        with open(legal_tokens_path, 'wb') as f:
            pickle.dump(self.state_legal_tokens, f)
        
        
        training_history_path = os.path.join(model_dir, 'training_history.json')
        with open(training_history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        
        metadata = {
            'vocab_size': self.vocab_size,
            'num_states': len(self.state_transitions),
            'state_types': self.state_types
        }
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model saved to: {model_dir}")
    
    def load(self, model_dir: str):
       
        state_transitions_path = os.path.join(model_dir, 'state_transitions.pkl')
        if os.path.exists(state_transitions_path):
            with open(state_transitions_path, 'rb') as f:
                self.state_transitions = defaultdict(lambda: defaultdict(int), pickle.load(f))
        
        
        state_counts_path = os.path.join(model_dir, 'state_counts.pkl')
        if os.path.exists(state_counts_path):
            with open(state_counts_path, 'rb') as f:
                self.state_counts = defaultdict(int, pickle.load(f))
        
        
        legal_tokens_path = os.path.join(model_dir, 'state_legal_tokens.pkl')
        if os.path.exists(legal_tokens_path):
            with open(legal_tokens_path, 'rb') as f:
                self.state_legal_tokens = pickle.load(f)
        
        
        training_history_path = os.path.join(model_dir, 'training_history.json')
        if os.path.exists(training_history_path):
            with open(training_history_path, 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)
        
        logger.info(f"Model loaded from {model_dir}")
    
    def validate_token(self, token: str, parse_state: ParseState) -> bool:
       
        token_id = self.vocabulary.get(token, self.vocabulary.get('[UNK]', 2))
        state_id = parse_state.get_state_id()
        legal_tokens = self.state_legal_tokens.get(state_id, set())
        return token_id in legal_tokens
    
    def plot_training_curves(self, save_path: str = None):
        
        if self.training_history.get('files_processed'):
            
            files = self.training_history['files_processed']
            x_label = 'Files processed'
            title_suffix = ''
        elif self.training_history.get('epoch'):
            
            files = self.training_history['epoch']
            x_label = 'Epochs'
            title_suffix = ' (by epoch)'
        else:
            logger.warning("No training history available, cannot plot curves")
            return
        
        
        fig, axes = plt.subplots(2, 3, figsize=(22, 10))
        fig.suptitle(f'Training Curves{title_suffix}', fontsize=16, fontweight='bold')
        
        epochs = files
        
        
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.training_history['states_count'], 'b-', linewidth=2, marker='o', markersize=6)
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel('States', fontsize=12)
        ax1.set_title('States Count', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain')
        
        
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.training_history['transitions_count'], 'g-', linewidth=2, marker='s', markersize=6)
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel('Transitions', fontsize=12)
        ax2.set_title('Transitions Count', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='plain')
        
        
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.training_history['vocab_size'], 'r-', linewidth=2, marker='^', markersize=6)
        ax3.set_xlabel(x_label, fontsize=12)
        ax3.set_ylabel('Vocabulary Size', fontsize=12)
        ax3.set_title('Vocabulary Size', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='plain')
        
        
        ax4 = axes[1, 0]
        if any(count > 0 for count in self.training_history['legal_tokens_count']):
            ax4.plot(epochs, self.training_history['legal_tokens_count'], 'm-', linewidth=2, marker='d', markersize=6)
            ax4.set_xlabel(x_label, fontsize=12)
            ax4.set_ylabel('Total Legal Tokens', fontsize=12)
            ax4.set_title('Legal Tokens Count', fontsize=13, fontweight='bold')
        else:
            
            total_legal = sum(len(tokens) for tokens in self.state_legal_tokens.values())
            ax4.bar(['Total Legal Tokens'], [total_legal], color='m', alpha=0.7)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.set_title('Final Legal Token Stats', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='plain')
        
        
        ax5 = axes[1, 1]
        if self.training_history.get('val_accuracy'):
            ax5.plot(epochs, self.training_history['val_accuracy'], 'c-', linewidth=2, marker='o', markersize=6)
            ax5.set_xlabel(x_label, fontsize=12)
            ax5.set_ylabel('Validation Accuracy', fontsize=12)
            ax5.set_title('Validation Accuracy (Legal-Token Inclusion)', fontsize=13, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim(0.0, 1.0)
        
        
        ax6 = axes[1, 2]
        if self.training_history.get('state_stability'):
            ax6.plot(epochs, self.training_history['state_stability'], 'k-', linewidth=2, marker='s', markersize=6, label='Stability (Jaccard)')
        if self.training_history.get('coverage_new_ratio'):
            ax6.plot(epochs, self.training_history['coverage_new_ratio'], 'y-', linewidth=2, marker='^', markersize=6, label='New-Pattern Ratio')
        ax6.set_xlabel(x_label, fontsize=12)
        ax6.set_ylabel('Value', fontsize=12)
        ax6.set_title('Stability & New-Pattern Ratio', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0.0, 1.05)
        ax6.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to: {save_path}")
        else:
            plt.savefig('/root/workspace/lzc/SynDef/training_curves.png', dpi=300, bbox_inches='tight')
            logger.info("Training curves saved to: training_curves.png")
        
        plt.close()
    
    def print_training_summary(self):
        
        if self.training_history.get('files_processed'):
            
            if not self.training_history['files_processed']:
                return
            logger.info("=" * 60)
            logger.info("Training Summary")
            logger.info("=" * 60)
            logger.info(f"Total files processed: {self.training_history['files_processed'][-1]}")
            logger.info(f"Final states: {self.training_history['states_count'][-1]}")
            logger.info(f"Final transitions: {self.training_history['transitions_count'][-1]}")
            logger.info(f"Vocabulary size: {self.training_history['vocab_size'][-1]}")
            if self.training_history['legal_tokens_count']:
                total_legal = sum(len(tokens) for tokens in self.state_legal_tokens.values())
                logger.info(f"Total legal tokens: {total_legal}")
            logger.info("=" * 60)
        elif self.training_history.get('epoch'):
            
            epochs = self.training_history['epoch']
            logger.info("=" * 60)
            logger.info("Training Summary")
            logger.info("=" * 60)
            logger.info(f"Epochs: {len(epochs)}")
            if self.training_history.get('total_files_processed'):
                logger.info(f"Total files processed: {self.training_history['total_files_processed'][-1]}")
            logger.info(f"Final states: {self.training_history['states_count'][-1]}")
            logger.info(f"Final transitions: {self.training_history['transitions_count'][-1]}")
            logger.info(f"Vocabulary size: {self.training_history['vocab_size'][-1]}")
            if self.training_history['legal_tokens_count']:
                total_legal = sum(len(tokens) for tokens in self.state_legal_tokens.values())
                logger.info(f"Total legal tokens: {total_legal}")
            if self.training_history.get('val_accuracy'):
                logger.info(f"Final validation accuracy: {self.training_history['val_accuracy'][-1]*100:.2f}%")
            if self.training_history.get('state_stability'):
                logger.info(f"Final state stability (Jaccard): {self.training_history['state_stability'][-1]*100:.2f}%")
            if self.training_history.get('coverage_new_ratio'):
                logger.info(f"Final new-pattern ratio: {self.training_history['coverage_new_ratio'][-1]*100:.2f}%")
            
            
            if len(epochs) > 1:
                logger.info("\nEpoch-wise training statistics:")
                for i, epoch in enumerate(epochs):
                    line = (f"  Epoch {epoch}: states={self.training_history['states_count'][i]}, "
                            f"transitions={self.training_history['transitions_count'][i]}, "
                            f"vocab={self.training_history['vocab_size'][i]}")
                    if i < len(self.training_history.get('val_accuracy', [])):
                        line += f", val_acc={self.training_history['val_accuracy'][i]*100:.2f}%"
                    if i < len(self.training_history.get('state_stability', [])):
                        line += f", stability={self.training_history['state_stability'][i]*100:.2f}%"
                    if i < len(self.training_history.get('coverage_new_ratio', [])):
                        line += f", new_pattern={self.training_history['coverage_new_ratio'][i]*100:.2f}%"
                    logger.info(line)
            
            logger.info("=" * 60)
        else:
            logger.warning("No training history data")


def main():
    
    vocabulary_path = '/root/workspace/lzc/SynDef/transform-model/tokenizer_vocab.json'
    train_data_dir = '/root/workspace/lzc/SynDef/automat-traindata'
    model_dir = '/root/workspace/lzc/SynDef/automat-model'
    
    
    logger.info("Initializing syntax automaton model...")
    automaton = SyntaxAutomaton(vocabulary_path)
    
    
    logger.info("Starting training...")
    
    automaton.train(train_data_dir, epochs=20, min_count=1, log_interval=100, num_workers=8,
                    early_stopping_patience=3, convergence_threshold=0.01)
    
    
    automaton.print_training_summary()
    
    
    logger.info("Plotting training curves...")
    plot_path = os.path.join(model_dir, 'training_curves.png')
    automaton.plot_training_curves(save_path=plot_path)
    
    
    logger.info("Saving model...")
    automaton.save(model_dir)
    
    
    logger.info("Testing model output...")
    test_state = ParseState(['class', 'method', 'if'])
    legal_mask = automaton.predict_legal_tokens(test_state)
    state_info = automaton.get_parse_state(test_state)
    
    logger.info(f"Test state: {state_info}")
    logger.info(f"Legal token mask shape: {legal_mask.shape}")
    logger.info(f"Number of legal tokens: {np.sum(legal_mask)}")
    
    logger.info("Done!")


if __name__ == '__main__':
    main()