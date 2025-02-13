import os
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from retriever.rank_bm25_retriever import RankBM25Retriever 
from retriever.faiss_retriever import FaissRetriever

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
        # 对documents进行分词
        tokenized_docs = [doc.split() for doc in documents]
        retriever = RankBM25Retriever(tokenized_docs, raw_documents)
        
        # 保存索引
        index_path = os.path.join(self.index_dir, "bm25.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(retriever, f)
        print(f"BM25 index saved to {index_path}")

    def build_faiss_index(self, documents, raw_documents):
        """构建FAISS索引"""
        retriever = FaissRetriever(documents, raw_documents)
        
        # 保存索引和原始文档
        faiss.write_index(retriever.index, 
                         os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.index_dir, "faiss_docs.pkl"), 'wb') as f:
            pickle.dump({
                "raw_docs": raw_documents,
                "dimension": retriever.dimension
            }, f)
        print(f"FAISS index saved to {self.index_dir}")

    def build_all_indexes(self, json_path):
        """构建所有类型的索引"""
        print(f"Loading data from {json_path}")
        documents, raw_documents = self.load_data(json_path)
        print(f"Loaded {len(documents)} documents")
        
        print("Building BM25 index...")
        self.build_bm25_index(documents, raw_documents)
        
        print("Building FAISS index...")
        self.build_faiss_index(documents, raw_documents)
        
        print("All indexes built successfully!")

if __name__ == "__main__":
    builder = IndexBuilder()
    json_path = "/home/hhl/rag_test/rag_demo/data/decompressed_files/wiki_01.json"
    builder.build_all_indexes(json_path)