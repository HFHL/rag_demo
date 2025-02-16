import os
import json
import pickle
import psutil
from pathlib import Path
import numpy as np
import gc
import faiss
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from datetime import datetime
import logging
from typing import List, Dict, Set, Any
from retriever.bm25s_retriever import BM25SRetriever
from retriever.rank_bm25_retriever import RankBM25Retriever
from retriever.faiss_retriever import FaissRetriever
import shutil

class IndexBuilder:
    def __init__(self, index_dir="./indexes", batch_size=1000):
        """
        初始化索引构建器
        Args:
            index_dir: 索引保存目录
            batch_size: 每批处理的文档数量
        """
        self.batch_size = batch_size
        # 转换为绝对路径
        self.index_dir = os.path.abspath(index_dir)
        
        # 确保索引目录存在
        os.makedirs(self.index_dir, exist_ok=True)
        
        # 设置日志和进度记录相关的路径
        self.processed_files_path = os.path.join(index_dir, "processed_files.txt")
        self.log_path = os.path.join(index_dir, "build_index.log")
        
        # 初始化日志
        self.setup_logging()
        
        # 加载已处理文件列表
        self.processed_files = self._load_processed_files()
        self.current_batch = 0
        self.checkpoint_path = os.path.join(index_dir, "checkpoint.json")
        self.load_checkpoint()
        self.bm25_path = os.path.join(self.index_dir, "bm25.pkl")
        self.bm25s_path = os.path.join(self.index_dir, "bm25s")
        self.initialize_indexes()
    
    def setup_logging(self):
        """设置日志记录"""
        # 确保日志目录存在
        log_dir = os.path.dirname(self.log_path)
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志初始化完成，日志文件: {self.log_path}")
    
    def _load_processed_files(self) -> Set[str]:
        """加载已处理文件列表"""
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as f:
                return set(line.strip() for line in f)
        return set()
    
    def _save_processed_file(self, filepath: str):
        """记录已处理的文件"""
        with open(self.processed_files_path, 'a') as f:
            f.write(f"{filepath}\n")
        self.processed_files.add(filepath)

    def log_memory_usage(self):
        """记录内存使用情况"""
        process = psutil.Process()
        mem_info = process.memory_info()
        self.logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

    def load_data(self, json_path):
        """使用迭代器方式加载和预处理JSON数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            raw_documents = []
            doc_count = 0
            
            for doc in data:
                if doc_count >= self.batch_size:
                    yield documents, raw_documents
                    documents = []
                    raw_documents = []
                    doc_count = 0
                    gc.collect()  # 强制垃圾回收
                
                # 处理title
                title = doc['title'].lower()
                title_doc = {'id': doc['id'], 'type': 'title', 'text': title}
                documents.append(title)
                raw_documents.append(title_doc)
                doc_count += 1
                
                # 处理每个段落文本
                for paragraph in doc['text']:
                    if not paragraph:
                        continue
                    
                    text_content = ' '.join(paragraph)
                    text_content = text_content.replace('<a href="', '').replace('">', ' ').replace('</a>', '')
                    processed_text = ' '.join(text_content.lower().split())
                    
                    if (processed_text.strip()):
                        paragraph_doc = {
                            'id': doc['id'],
                            'type': 'paragraph',
                            'text': processed_text,
                            'title': title
                        }
                        documents.append(processed_text)
                        raw_documents.append(paragraph_doc)
                        doc_count += 1
                        
                        if doc_count >= self.batch_size:
                            yield documents, raw_documents
                            documents = []
                            raw_documents = []
                            doc_count = 0
                            gc.collect()
            
            if documents:  # 处理剩余的文档
                yield documents, raw_documents
                
        except Exception as e:
            self.logger.error(f"Error loading file {json_path}: {str(e)}")
            return [], []

    def process_single_json(self, json_path: str) -> List[tuple]:
        """处理单个JSON文件，返回多个批次的结果"""
        try:
            if (json_path in self.processed_files):
                self.logger.info(f"跳过已处理的文件: {json_path}")
                return []
            
            self.logger.info(f"处理文件: {json_path}")
            self.log_memory_usage()
            
            results = []
            for batch_docs, batch_raw_docs in self.load_data(json_path):
                if batch_docs and batch_raw_docs:
                    results.append((batch_docs, batch_raw_docs))
                
            self._save_processed_file(json_path)
            return results
            
        except Exception as e:
            self.logger.error(f"处理文件 {json_path} 时出错: {str(e)}")
            return []

    def find_all_json_files(self, data_dir: str) -> List[str]:
        """查找目录下所有JSON文件"""
        # 转换为绝对路径
        data_dir = os.path.abspath(data_dir)
        self.logger.info(f"搜索目录的绝对路径: {data_dir}")
        
        if not os.path.exists(data_dir):
            self.logger.error(f"目录不存在: {data_dir}")
            return []
            
        json_files = []
        # 使用os.walk遍历所有子目录
        for root, dirs, files in os.walk(data_dir):
            self.logger.debug(f"当前搜索目录: {root}")
            self.logger.debug(f"包含的子目录: {dirs}")
            self.logger.debug(f"当前目录下的文件数: {len(files)}")
            
            for file in files:
                if file.endswith('.json'):
                    full_path = os.path.join(root, file)
                    json_files.append(full_path)
                    self.logger.debug(f"找到JSON文件: {full_path}")
        
        self.logger.info(f"共找到 {len(json_files)} 个JSON文件")
        # 打印前几个文件路径作为示例
        if json_files:
            self.logger.info("示例文件:")
            for f in json_files[:3]:
                self.logger.info(f"  - {f}")
        
        return json_files

    def load_checkpoint(self):
        """加载检查点信息"""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = {
                'current_batch': 0,
                'processed_count': 0
            }

    def save_checkpoint(self):
        """保存检查点信息"""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f)

    def initialize_indexes(self):
        """初始化或加载现有索引"""
        try:
            if os.path.exists(self.bm25_path):
                self.bm25_index = RankBM25Retriever.load(self.bm25_path)
            else:
                self.bm25_index = None

            if os.path.exists(self.bm25s_path):
                self.bm25s_index = BM25SRetriever.load(self.bm25s_path)
            else:
                self.bm25s_index = None
                
        except Exception as e:
            self.logger.error(f"Error initializing indexes: {str(e)}")
            self.bm25_index = None
            self.bm25s_index = None

    def append_to_index(self, documents, raw_documents, index_type):
        """将文档追加到现有索引"""
        try:
            if index_type == 'bm25':
                tokenized_docs = [doc.split() for doc in documents]
                if self.bm25_index is None:
                    # 首次创建索引
                    self.bm25_index = RankBM25Retriever(tokenized_docs, raw_documents)
                else:
                    # 追加到现有索引
                    self.bm25_index.add_documents(tokenized_docs, raw_documents)
                
                # 保存更新后的索引
                try:
                    self.bm25_index.save(self.bm25_path)
                    self.logger.info(f"Successfully saved BM25 index to {self.bm25_path}")
                except Exception as save_error:
                    self.logger.error(f"Error saving BM25 index: {str(save_error)}")
                    raise
                    
            elif index_type == 'bm25s':
                try:
                    if self.bm25s_index is None:
                        # 首次创建索引
                        self.bm25s_index = BM25SRetriever(documents, raw_documents)
                    else:
                        # 追加到现有索引
                        self.bm25s_index.add_documents(documents, raw_documents)
                    
                    # 保存更新后的索引
                    self.bm25s_index.save(self.bm25s_path)
                    self.logger.info(f"Successfully saved BM25S index to {self.bm25s_path}")
                except Exception as save_error:
                    self.logger.error(f"Error with BM25S index: {str(save_error)}")
                    raise
                
            self.logger.info(f"Updated {index_type} index with {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Error updating {index_type} index: {str(e)}")
            raise

    def process_and_save_batch(self, documents, raw_documents):
        """处理一个批次的文档并追加到索引"""
        try:
            self.logger.info(f"Processing batch {self.current_batch} with {len(documents)} documents")
            
            # 追加到BM25索引
            self.append_to_index(documents, raw_documents, 'bm25')
            
            # 追加到BM25S索引
            self.append_to_index(documents, raw_documents, 'bm25s')
            
            # 更新检查点
            self.checkpoint['current_batch'] = self.current_batch
            self.checkpoint['processed_count'] += len(documents)
            self.save_checkpoint()
            
            self.current_batch += 1
            gc.collect()
            self.log_memory_usage()
            
        except Exception as e:
            self.logger.error(f"Error processing batch {self.current_batch}: {str(e)}")
            raise
    
    def merge_document_lists(self, all_results: List[List[tuple]]) -> None:
        """分批处理并保存结果"""
        all_documents = []
        all_raw_documents = []
        
        for batch_results in all_results:
            if not batch_results:
                continue
                
            for batch_data in batch_results:
                if not isinstance(batch_data, tuple) or len(batch_data) != 2:
                    self.logger.warning(f"跳过无效的批次数据: {batch_data}")
                    continue
                    
                docs, raw_docs = batch_data
                all_documents.extend(docs)
                all_raw_documents.extend(raw_docs)
                
                # 当累积的文档数量达到批处理大小时进行处理
                if len(all_documents) >= self.batch_size:
                    self.process_and_save_batch(all_documents, all_raw_documents)
                    all_documents = []
                    all_raw_documents = []
        
        # 处理剩余的文档
        if all_documents:
            self.process_and_save_batch(all_documents, all_raw_documents)

    def merge_all_indexes(self):
        """合并所有批次的索引（如果需要）"""
        # 这里可以添加合并索引的逻辑
        pass

    def build_all_indexes(self, data_dir: str, max_workers: int = 4):
        """处理目录下所有JSON文件并构建索引"""
        start_time = datetime.now()
        self.logger.info(f"开始处理目录: {data_dir}")
        
        # 查找所有JSON文件
        json_files = self.find_all_json_files(data_dir)
        total_files = len(json_files)
        self.logger.info(f"找到 {total_files} 个JSON文件")
        
        # 确保索引目录存在且清空旧索引
        if os.path.exists(self.bm25_path):
            os.remove(self.bm25_path)
        if os.path.exists(self.bm25s_path):
            shutil.rmtree(self.bm25s_path, ignore_errors=True)
            
        # 并行处理文件
        all_results = []
        total_docs = 0
        with tqdm(total=total_files, desc="处理JSON文件") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_json, json_file): json_file 
                    for json_file in json_files
                }
                
                for future in as_completed(futures):
                    file = futures[future]
                    try:
                        batch_results = future.result()
                        if batch_results:
                            # 直接处理这个文件的结果，而不是收集所有结果
                            self.merge_document_lists([batch_results])
                            doc_count = sum(len(docs) for batch in batch_results 
                                         if isinstance(batch, tuple) and len(batch) == 2 
                                         for docs in [batch[0]])
                            total_docs += doc_count
                            
                        pbar.update(1)
                        self.logger.info(f"已处理文档数: {total_docs}")
                        self.log_memory_usage()
                    except Exception as e:
                        self.logger.error(f"处理文件 {file} 失败: {str(e)}")
                        pbar.update(1)
        
        # 可选：合并所有批次的索引
        # self.merge_all_indexes()
        
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"索引构建完成! 总用时: {duration}")

if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构造相对于脚本的路径
    data_dir = os.path.join(current_dir, "", "decompressed_files")
    index_dir = os.path.join(current_dir, "indexes")
    
    builder = IndexBuilder(index_dir=index_dir, batch_size=1000)
    builder.logger.info(f"当前工作目录: {os.getcwd()}")
    builder.logger.info(f"数据目录: {data_dir}")
    
    # 设置并行进程数
    builder.build_all_indexes(data_dir, max_workers=4)