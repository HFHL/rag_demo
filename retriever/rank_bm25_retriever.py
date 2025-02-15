from .retriever import Retriever
from rank_bm25 import BM25Okapi
import pickle
import os

class RankBM25Retriever(Retriever):
    def __init__(self, corpus, raw_docs):
        """
        Initialize BM25 retriever
        Args:
            corpus: List of tokenized documents
            raw_docs: Original documents with metadata
        """
        self.bm25 = BM25Okapi(corpus)
        self.raw_docs = raw_docs
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve most relevant documents
        Args:
            query: Query string
            top_k: Number of documents to return
        Returns:
            List of formatted documents
        """
        # 对查询进行分词
        tokenized_query = query.lower().split()
        
        # 获取文档得分和索引
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        # 返回原始文档
        results = []
        for idx in top_n:
            doc = self.raw_docs[idx]
            
            if doc.get('type') == 'title':
                # 如果是标题文档
                result = f"Title: {doc['text']}"
            elif doc.get('type') == 'paragraph':
                # 如果是段落文档
                result = f"Title: {doc['title']}\nContent: {doc['text']}"
            else:
                # 向后兼容,处理旧格式文档
                result = str(doc)
                
            results.append(result)
            
        return results

    def save(self, path: str):
        """
        保存检索器到文件
        Args:
            path: 保存路径
        """
        save_dict = {
            'bm25': self.bm25,
            'corpus': self.corpus,
            'raw_docs': self.raw_docs
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(self, path: str):
        """
        从文件加载检索器
        Args:
            path: 加载路径
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        self.bm25 = save_dict['bm25']
        self.corpus = save_dict['corpus']
        self.raw_docs = save_dict['raw_docs']
