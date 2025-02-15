import os
import pickle
from typing import List, Tuple
import numpy as np
import bm25s
import Stemmer
from .retriever import Retriever

class BM25Retriever:
    def retrieve(self, question):
        # 这里实现BM25检索逻辑
        return ["相关文档1", "相关文档2"]

class BM25SRetriever(Retriever):
    def __init__(self, texts: List[str] = None, raw_docs: List[dict] = None):
        """
        初始化BM25S检索器
        Args:
            texts: 文档文本列表
            raw_docs: 原始文档列表(包含metadata)
        """
        self.raw_docs = raw_docs
        self.texts = texts
        self.stemmer = Stemmer.Stemmer('english')
        self.retriever = bm25s.BM25()
        
        if texts is not None:
            self._build_index(texts)
    
    def _build_index(self, texts: List[str]):
        """构建索引"""
        # 对文档进行分词和stemming
        corpus_tokens = bm25s.tokenize(texts, 
                                     stopwords="en",
                                     stemmer=self.stemmer)
        # 构建索引
        self.retriever.index(corpus_tokens)
        self.corpus_tokens = corpus_tokens

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        检索相关文档
        Args:
            query: 查询字符串
            top_k: 返回前k个结果
        Returns:
            检索到的文档列表
        """
        # 对查询进行分词和stemming
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        
        # 检索
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)
        
        # 返回原始文档
        retrieved_docs = []
        for i in range(results.shape[1]):
            doc_id = results[0, i]
            doc = self.raw_docs[doc_id]
            if doc.get('type') == 'title':
                result = f"Title: {doc['text']}"
            elif doc.get('type') == 'paragraph':
                result = f"Title: {doc['title']}\nContent: {doc['text']}"
            else:
                result = str(doc)
            retrieved_docs.append(result)
            
        return retrieved_docs

    def save(self, path: str):
        """
        保存检索器
        Args:
            path: 保存路径
        """
        # 创建目录
        os.makedirs(path, exist_ok=True)
        
        # 保存BM25S模型
        self.retriever.save(os.path.join(path, "bm25s_model"))
        
        # 保存其他数据
        with open(os.path.join(path, "retriever_data.pkl"), "wb") as f:
            pickle.dump({
                "raw_docs": self.raw_docs,
                "texts": self.texts,
                "corpus_tokens": self.corpus_tokens
            }, f)

    def load(self, path: str):
        """
        加载检索器
        Args:
            path: 加载路径
        """
        # 加载BM25S模型
        self.retriever = bm25s.BM25.load(os.path.join(path, "bm25s_model"))
        
        # 加载其他数据
        with open(os.path.join(path, "retriever_data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.raw_docs = data["raw_docs"]
            self.texts = data["texts"] 
            self.corpus_tokens = data["corpus_tokens"]
