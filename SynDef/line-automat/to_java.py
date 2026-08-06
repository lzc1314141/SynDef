import os
import pandas as pd
from pathlib import Path

def convert_path_to_filename(file_path):
    
    return file_path.replace('.java', '').replace('/', '_').replace('\\', '_') + '.java'

def extract_version_from_filename(filename):
    
    if '_ground' in filename:
        return filename.split('_ground')[0]
    return filename.replace('.csv', '')

def extract_java_files_from_csv():
    
    file_level_dir = "/root/workspace/lzc/automat/sourcefile/File-level"
    target_base = "/root/workspace/lzc/automat/sourcefile/sourcedata"
    
    os.makedirs(target_base, exist_ok=True)
    
    total_files = 0
    total_versions = 0
    
    print("="*80)
    print("开始从File-level表格中提取Java文件")
    print("="*80)
    
    csv_files = [f for f in os.listdir(file_level_dir) if f.endswith('.csv')]
    
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(file_level_dir, csv_file)
        
        version = extract_version_from_filename(csv_file)
        print(f"\n处理版本: {version} (文件: {csv_file})")
        
        version_target_dir = os.path.join(target_base, version)
        os.makedirs(version_target_dir, exist_ok=True)
        
        try:
            df = None
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"警告: 无法读取CSV文件 {csv_file}，尝试了多种编码格式")
                continue
            
            if 'File' not in df.columns or 'SRC' not in df.columns:
                print(f"警告: CSV文件 {csv_file} 缺少必要的列 (File, SRC)")
                continue
            
            file_count = 0
            
            for index, row in df.iterrows():
                try:
                    file_path = row['File']
                    source_code = row['SRC']
                    
                    if pd.isna(source_code) or not isinstance(source_code, str):
                        print(f"  跳过非字符串源代码: {file_path} (类型: {type(source_code)})")
                        continue
                    
                    java_filename = convert_path_to_filename(file_path)
                    java_file_path = os.path.join(version_target_dir, java_filename)
                    
                    with open(java_file_path, 'w', encoding='utf-8') as f:
                        f.write(source_code)
                    
                    file_count += 1
                    total_files += 1
                    
                    if file_count % 100 == 0:
                        print(f"  已处理 {file_count} 个文件...")
                        
                except Exception as e:
                    print(f"  处理行 {index} 时出错: {e}")
                    continue
            
            print(f"版本 {version} 完成，共生成 {file_count} 个Java文件")
            total_versions += 1
            
        except Exception as e:
            print(f"处理CSV文件 {csv_file} 时出错: {e}")
    
    print(f"\n{'='*80}")
    print("提取完成！")
    print(f"{'='*80}")
    print(f"总共处理了 {total_versions} 个版本")
    print(f"总共生成了 {total_files} 个Java文件")
    print(f"文件保存在: {target_base}")
    
    if os.path.exists(target_base):
        print(f"\n生成的版本目录:")
        for item in sorted(os.listdir(target_base)):
            item_path = os.path.join(target_base, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) if f.endswith('.java')])
                print(f"  - {item}/ ({file_count} 个Java文件)")

if __name__ == "__main__":
    extract_java_files_from_csv()