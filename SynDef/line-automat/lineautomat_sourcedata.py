#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from collections import defaultdict

class LineAutomaton:
    """为单行代码创建自动机的类"""
    
    def __init__(self, line_content, file_name, line_number): 
        self.line_content = line_content.strip()  
        self.file_name = file_name   
        self.line_number = line_number  
        self.states = []  
        self.transitions = []  
        self.start_state = "q0"  
        self.accept_states = []  
        
    def tokenize_line(self):
        """将代码行分解为单词token"""
        if not self.line_content or self.line_content.strip() == '':  
            return []
        
        line = re.sub(r'//.*$', '', self.line_content)  
        line = re.sub(r'/\*.*?\*/', '', line)  
        
        if not line.strip():
            return []
        
        
        tokens = re.findall(r'\b\w+\b|[+\-*/=<>!&|]+|[(){}\[\];,.]|"[^"]*"|\'[^\']*\'', line)
        
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens
    
    def build_automaton(self):
        tokens = self.tokenize_line()  
        
        if not tokens:
            self.states = [self.start_state]
            self.accept_states = [self.start_state]
            return
        
        num_states = len(tokens) + 1   
        self.states = [f"q{i}" for i in range(num_states)]  
        
        self.accept_states = [f"q{len(tokens)}"]  
        
        for i, token in enumerate(tokens):  
            transition = {  
                "from_state": f"q{i}",  
                "to_state": f"q{i+1}",  
                "symbol": token,  
                "token_index": i  
            }
            self.transitions.append(transition)  
    
    def to_dict(self):
        return {
            "metadata": {
                "source_file": self.file_name,
                "line_number": self.line_number,
                "line_content": self.line_content,
                "created_from": "Java source code line"
            },
            "automaton": {
                "states": self.states,
                "start_state": self.start_state,
                "accept_states": self.accept_states,
                "transitions": self.transitions,
                "alphabet": list(set(t["symbol"] for t in self.transitions))  #字母表：转移表中的所有符号
            }
        }

def create_line_automata_for_version(version_name, source_dir, output_dir):
    
    
    if not os.path.exists(source_dir):
        print(f"警告: 源代码目录不存在 - {source_dir}")
        return 0, 0, 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = 0
    total_lines = 0
    total_automata = 0
    empty_lines = 0
    
    all_automata = []
    
    for filename in os.listdir(source_dir):
        if not filename.endswith('.java'):  
            continue
        
        source_file = os.path.join(source_dir, filename) 
        total_files += 1
        
        
        try:
            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:  
                lines = f.readlines()  
            
            file_automata = [] 
            
            for line_num, line_content in enumerate(lines, 1):  
                total_lines += 1
                
                automaton = LineAutomaton(line_content, filename, line_num)  
                automaton.build_automaton()  
                
                if automaton.transitions or line_content.strip():
                    automaton_dict = automaton.to_dict()
                    file_automata.append(automaton_dict) 
                    all_automata.append(automaton_dict)  
                    total_automata += 1
                else:
                    empty_lines += 1
            
            
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    output_file = os.path.join(output_dir, f"{version_name}.json")
    automata_data = {
        "project_name": version_name,
        "processing_summary": {
            "total_files_processed": total_files, 
            "total_lines_processed": total_lines, 
            "total_automata_generated": total_automata, 
            "empty_lines_skipped": empty_lines, 
            "source_directory": source_dir, 
            "output_directory": output_dir 
        },
        "automaton_structure_info": {
            "description": "每个自动机代表一行Java代码",
            "states": "状态编号为q0, q1, ..., qn",
            "transitions": "每个单词/token作为转移条件",
            "start_state": "q0",
            "accept_state": "最后一个状态"
        },
        "automata": all_automata
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(automata_data, f, indent=2, ensure_ascii=False)
    
    

    return total_files, total_lines, total_automata, empty_lines

def create_line_automata():
    
    cleanfile_dir = '/root/workspace/lzc/automat/cleanfile_sourcedata'
    output_base_dir = '/root/workspace/lzc/automat/Line-auto_sourcedata'
    
    
    if not os.path.exists(cleanfile_dir):
        print(f"错误: 清理文件目录不存在 - {cleanfile_dir}")
        return
    
    version_dirs = [d for d in os.listdir(cleanfile_dir) 
                   if os.path.isdir(os.path.join(cleanfile_dir, d))]
    
    


    total_processed_files = 0
    total_processed_lines = 0
    total_generated_automata = 0
    total_skipped_lines = 0
    successful_versions = 0
    
    for version in sorted(version_dirs):
        try:
            source_dir = os.path.join(cleanfile_dir, version)
            output_dir = os.path.join(output_base_dir, version)
            
            files, lines, automata, skipped = create_line_automata_for_version(
                version, source_dir, output_dir)
            
            if files > 0:  
                total_processed_files += files
                total_processed_lines += lines
                total_generated_automata += automata
                total_skipped_lines += skipped
                successful_versions += 1
                
        except Exception as e:
            print(f"处理版本 {version} 时出错: {str(e)}")
    
    


if __name__ == "__main__":
    create_line_automata()