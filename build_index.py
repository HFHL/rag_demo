import os
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from retriever.rank_bm25_retriever import RankBM25Retriever 
from retriever.faiss_retriever import FaissRetriever
from retriever.bm25s_retriever import BM25SRetriever

class IndexBuilder:
    def __init__(self, index_dir="/home/hhl/rag_test/rag_demo/indexes"):
        """
        初始化索引构建器
        Args:
            index_dir: 索引保存目录
        """
        self.index_dir = index_dir
        # 确保索引目录存在
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        
    def load_data(self, json_path):
        """加载和预处理JSON数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        raw_documents = []
        
        for doc in data:
            # 处理title
            title = doc['title'].lower()
            title_doc = {'id': doc['id'], 'type': 'title', 'text': title}
            documents.append(title)
            raw_documents.append(title_doc)
            
            # 处理每个段落文本
            for paragraph in doc['text']:
                if not paragraph:  # 跳过空段落
                    continue
                # 合并段落中的句子
                text_content = ' '.join(paragraph)
                # 移除HTML标签
                text_content = text_content.replace('<a href="', '').replace('">', ' ').replace('</a>', '')
                # 规范化处理
                processed_text = ' '.join(text_content.lower().split())
                
                if processed_text.strip():  # 只添加非空文本
                    paragraph_doc = {
                        'id': doc['id'],
                        'type': 'paragraph',
                        'text': processed_text,
                        'title': title  # 保留原文标题用于上下文
                    }
                    documents.append(processed_text)
                    raw_documents.append(paragraph_doc)
            
        return documents, raw_documents

    def build_bm25_index(self, documents, raw_documents):
        """构建BM25索引"""
        print("Tokenizing documents for BM25...")
        tokenized_docs = []
        for doc in tqdm(documents, desc="BM25 Tokenization"):
            tokenized_docs.append(doc.split())
            
        print("Building BM25 index...")
        retriever = RankBM25Retriever(tokenized_docs, raw_documents)
        
        print("Saving BM25 index...")
        index_path = os.path.join(self.index_dir, "bm25.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(retriever, f)
        print(f"BM25 index saved to {index_path}")

    def build_faiss_index(self, documents, raw_documents):
        """构建FAISS索引"""
        print("Building FAISS index...")
        retriever = FaissRetriever(documents, raw_documents)
        
        print("Saving FAISS index...")
        # 保存索引和原始文档
        faiss.write_index(retriever.index, 
                         os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.index_dir, "faiss_docs.pkl"), 'wb') as f:
            pickle.dump({
                "raw_docs": raw_documents,
                "dimension": retriever.dimension
            }, f)
        print(f"FAISS index saved to {self.index_dir}")

    def build_bm25s_index(self, documents, raw_documents):
        """构建BM25S索引"""
        print("Building BM25S index...")
        retriever = BM25SRetriever(documents, raw_documents)
        
        print("Saving BM25S index...")
        index_path = os.path.join(self.index_dir, "bm25s")
        retriever.save(index_path)
        print(f"BM25S index saved to {index_path}")

    def build_all_indexes(self, json_path):
        """构建所有类型的索引"""
        print(f"Loading data from {json_path}")
        documents, raw_documents = self.load_data(json_path)
        print(f"Loaded {len(documents)} documents")
        
        # 显示总进度
        total_steps = 3  # BM25, BM25S, FAISS
        with tqdm(total=total_steps, desc="Overall Progress") as pbar:
            print("\nBuilding BM25 index...")
            self.build_bm25_index(documents, raw_documents)
            pbar.update(1)
            
            print("\nBuilding BM25S index...")
            self.build_bm25s_index(documents, raw_documents)
            pbar.update(1)
            
            # print("\nBuilding FAISS index...")
            # self.build_faiss_index(documents, raw_documents)
            # pbar.update(1)
        
        print("\nAll indexes built successfully!")

if __name__ == "__main__":
    builder = IndexBuilder()
    json_path = "/home/hhl/rag_test/rag_demo/data/decompressed_files/wiki_01.json"
    builder.build_all_indexes(json_path)