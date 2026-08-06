#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
from collections import defaultdict

COMMENT_PATTERN_1 = re.compile(r'//.*$')   
COMMENT_PATTERN_2 = re.compile(r'/\*.*?\*/')   
TOKEN_PATTERN = re.compile(r'\b\w+\b|[+\-*/=<>!&|]+|[(){}\[\];,.]|"[^"]*"|\'[^\']*\'')   

def tokenize_java_line(line_content):   

    if not line_content or line_content.strip() == '':
        return []   
    line = COMMENT_PATTERN_1.sub('', line_content)
    line = COMMENT_PATTERN_2.sub('', line)
    
    if not line.strip():
        return []   
    
    tokens = TOKEN_PATTERN.findall(line)  
    tokens = [token.strip() for token in tokens if token.strip()]  
    
    return tokens   

def convert_path_to_filename(file_path):
    
    return file_path.replace('.java', '').replace('/', '_') + '.java' 

def clean_single_version(version_name, csv_file, source_dir, output_dir):
    
    
    if not os.path.exists(csv_file):
        print(f"警告: CSV文件不存在 - {csv_file}")
        return 0, 0, 0
        
    if not os.path.exists(source_dir):
        print(f"警告: 源代码目录不存在 - {source_dir}")
        return 0, 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    
    defective_info = defaultdict(set)  
    for _, row in df.iterrows():
        file_path = row['File'] 
        src_content = row['SRC'].strip()  
        filename = convert_path_to_filename(file_path) 
        
        tokens = tokenize_java_line(src_content) 
        if tokens:  
            token_tuple = tuple(tokens)  
            defective_info[filename].add(token_tuple)  
    
    
    all_defective_tokens = set()
    for token_sets in defective_info.values():
        all_defective_tokens.update(token_sets)
    
    processed_files = 0 
    total_deleted_lines = 0 
    
    for filename in os.listdir(source_dir): 
        if not filename.endswith('.java'): 
            continue
            
        source_file = os.path.join(source_dir, filename) 
        output_file = os.path.join(output_dir, filename) 
        
        try:
            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines() 
            
            original_count = len(lines) 
            deleted_count = 0 
            
            
            if all_defective_tokens:  
                has_deletions = False
                
                
                i = len(lines) - 1
                while i >= 0:
                    current_line = lines[i].rstrip('\n\r') 
                    
                    current_tokens = tokenize_java_line(current_line)
                    current_token_tuple = tuple(current_tokens) 
                    
                    if current_token_tuple in all_defective_tokens:
                        if not has_deletions:  
                            has_deletions = True 
                        del lines[i] 
                        deleted_count += 1 
                    
                    i -= 1 
                    
                if has_deletions:
                    print(f"    原始行数: {original_count}")
                    print(f"    删除行数: {deleted_count}")
                    print(f"    最终行数: {len(lines)}")
                
                total_deleted_lines += deleted_count 
            
            with open(output_file, 'w', encoding='utf-8') as f: 
                f.writelines(lines) 
            
            processed_files += 1
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    
    return processed_files, len(defective_info), total_deleted_lines

def clean_defective_lines():
    
    source_data_dir = '/root/workspace/lzc/automat/sourcefile/sourcedata'
    line_level_dir = '/root/workspace/lzc/automat/sourcefile/Line-level'
    output_base_dir = '/root/workspace/lzc/automat/cleanfile_sourcedata'
    
    if not os.path.exists(source_data_dir):
        print(f"错误: 源数据目录不存在 - {source_data_dir}")
        return
    
    version_dirs = [d for d in os.listdir(source_data_dir) 
                   if os.path.isdir(os.path.join(source_data_dir, d))] 
    
    print(f"发现 {len(version_dirs)} 个版本目录:")
    for version in sorted(version_dirs): 
        print(f"  - {version}") 
    
    total_processed_files = 0 
    total_defective_files = 0 
    total_deleted_lines = 0 
    successful_versions = 0 
    
    for version in sorted(version_dirs):
        try:
            source_dir = os.path.join(source_data_dir, version) 
            csv_file = os.path.join(line_level_dir, f"{version}_defective_lines_dataset.csv") 
            output_dir = os.path.join(output_base_dir, version) 
            
            processed, defective, deleted = clean_single_version(
                version, csv_file, source_dir, output_dir)
            
            if processed > 0:  
                total_processed_files += processed
                total_defective_files += defective
                total_deleted_lines += deleted
                successful_versions += 1
                
        except Exception as e:
            print(f"处理版本 {version} 时出错: {str(e)}")
    
if __name__ == "__main__":
    clean_defective_lines()