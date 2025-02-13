#  定义一个retriever类
from .retriever import Retriever
from rank_bm25 import BM25Okapi
import re

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

    def preprocess(self, text):
        """Simple preprocessing - lowercase and remove special chars"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

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
