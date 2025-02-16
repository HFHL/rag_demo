from rank_bm25 import BM25Okapi  # 替换为更可靠的BM25实现
import numpy as np
import json
import os

class BM25SRetriever:
    def __init__(self, documents=None, raw_documents=None):
        """初始化检索器"""
        self.documents = documents if documents else []
        self.raw_documents = raw_documents if raw_documents else []
        self.bm25 = None
        if documents:
            self._build_index()
            
    def _build_index(self):
        """构建BM25索引"""
        # 对文档进行分词
        tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def add_documents(self, new_documents, new_raw_documents):
        """添加新文档到索引"""
        if not self.documents:
            self.documents = new_documents
            self.raw_documents = new_raw_documents
        else:
            self.documents.extend(new_documents)
            self.raw_documents.extend(new_raw_documents)
        self._build_index()  # 重新构建索引
        
    def search(self, query, top_k=10):
        """搜索最相关的文档"""
        if not self.bm25:
            return []
            
        # 对查询进行分词
        tokenized_query = query.lower().split()
        
        # 获取文档得分
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top_k个最相关文档的索引
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            results.append({
                'score': float(doc_scores[idx]),  # 转换为Python float
                'document': self.documents[idx],
                'metadata': self.raw_documents[idx]
            })
            
        return results
        
    def save(self, path):
        """保存索引到文件"""
        os.makedirs(path, exist_ok=True)
        
        # 保存文档数据
        save_data = {
            'documents': self.documents,
            'raw_documents': self.raw_documents
        }
        
        save_path = os.path.join(path, "index_data.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False)
            
    @classmethod
    def load(cls, path):
        """从文件加载索引"""
        save_path = os.path.join(path, "index_data.json")
        
        if not os.path.exists(save_path):
            return cls()
            
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        instance = cls(
            documents=data['documents'],
            raw_documents=data['raw_documents']
        )
        
        return instance

    def get_document_count(self):
        """获取索引中的文档数量"""
        return len(self.documents)

    def get_statistics(self):
        """获取索引统计信息"""
        return {
            'document_count': len(self.documents),
            'has_index': self.bm25 is not None,
            'average_document_length': np.mean([len(doc.split()) for doc in self.documents]) if self.documents else 0
        }
