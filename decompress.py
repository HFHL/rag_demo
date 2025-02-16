import os
import bz2
import json
import shutil
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

def process_single_file(args: Tuple[str, str, str]) -> None:
    """处理单个bz2文件的worker函数"""
    bz2_path, root_dir, output_dir = args
    try:
        # 生成输出文件路径
        dirpath = os.path.dirname(bz2_path)
        filename = os.path.basename(bz2_path)
        relative_path = os.path.relpath(dirpath, root_dir)
        target_dir = os.path.join(output_dir, relative_path)
        
        os.makedirs(target_dir, exist_ok=True)
        
        # 设置输出文件路径
        base_name = filename[:-4]  # 移除.bz2
        if not base_name.endswith('.json'):
            base_name += '.json'
        output_path = os.path.join(target_dir, base_name)
        
        # 处理文件
        json_objects = []
        with bz2.open(bz2_path, 'rt', encoding='utf-8') as source:
            for line in source:
                try:
                    json_obj = json.loads(line.strip())
                    json_objects.append(json_obj)
                except json.JSONDecodeError as je:
                    print(f"Error parsing JSON line in {bz2_path}: {str(je)}")
                    continue
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as dest:
            json.dump(json_objects, dest, ensure_ascii=False, indent=2)
        
        # 删除原始文件
        os.remove(bz2_path)
        print(f"Processed: {bz2_path}")
        
    except Exception as e:
        print(f"Error processing {bz2_path}: {str(e)}")

def decompress_and_remove(root_dir: str, output_dir: str, batch_size: int = 10) -> None:
    # 转换为绝对路径
    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)
    
    print(f"Using absolute paths:")
    print(f"Root directory: {root_dir}")
    print(f"Output directory: {output_dir}")

    # 确保目标目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查源目录是否存在
    if not os.path.exists(root_dir):
        print(f"Error: Source directory does not exist: {root_dir}")
        return
    
    # 收集所有需要处理的文件
    bz2_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.bz2'):
                bz2_files.append(os.path.join(dirpath, filename))
    
    total_files = len(bz2_files)
    print(f"Found {total_files} .bz2 files to process")
    
    # 创建进程池
    num_processes = min(cpu_count(), batch_size)
    print(f"Using {num_processes} processes")
    
    # 准备参数
    process_args = [(f, root_dir, output_dir) for f in bz2_files]
    
    # 批量处理文件
    with Pool(processes=num_processes) as pool:
        for i in range(0, len(process_args), batch_size):
            batch = process_args[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} of {(total_files + batch_size - 1)//batch_size}")
            pool.map(process_single_file, batch)
            print(f"Completed batch {i//batch_size + 1}")

if __name__ == "__main__":
    root_directory = "./data/enwiki-20171001-pages-meta-current-withlinks-processed"
    output_directory = "./decompressed_files"
    # 设置批处理大小为20
    decompress_and_remove(root_directory, output_directory, batch_size=20)