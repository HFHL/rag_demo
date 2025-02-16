from rank_bm25 import BM25Okapi
import numpy as np
import pickle

class RankBM25Retriever:
    def __init__(self, tokenized_documents=None, raw_documents=None):
        """初始化检索器"""
        self.tokenized_documents = tokenized_documents if tokenized_documents else []
        self.raw_documents = raw_documents if raw_documents else []
        self.bm25 = None
        if tokenized_documents:
            self._build_index()
            
    def _build_index(self):
        """构建BM25索引"""
        self.bm25 = BM25Okapi(self.tokenized_documents)
        
    def _save_bm25_params(self):
        """保存BM25模型的参数"""
        if self.bm25 is not None:
            return {
                'idf': self.bm25.idf,
                'doc_len': self.bm25.doc_len,
                'avgdl': self.bm25.avgdl,
                'epsilon': self.bm25.epsilon
            }
        return None
        
    def _load_bm25_params(self, params):
        """从参数重建BM25模型"""
        if params and self.tokenized_documents:
            self._build_index()
            self.bm25.idf = params['idf']
            self.bm25.doc_len = params['doc_len']
            self.bm25.avgdl = params['avgdl']
            self.bm25.epsilon = params['epsilon']
            
    def add_documents(self, new_tokenized_docs, new_raw_docs):
        """添加新文档到索引"""
        self.tokenized_documents.extend(new_tokenized_docs)
        self.raw_documents.extend(new_raw_docs)
        self._build_index()
        
    def save(self, path):
        """保存检索器到文件"""
        save_data = {
            'tokenized_documents': self.tokenized_documents,
            'raw_documents': self.raw_documents,
            'bm25_params': self._save_bm25_params()
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
            
    @classmethod
    def load(cls, path):
        """从文件加载检索器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        instance = cls(
            tokenized_documents=data['tokenized_documents'],
            raw_documents=data['raw_documents']
        )
        instance._load_bm25_params(data['bm25_params'])
        return instance

    def search(self, tokenized_query, top_k=10):
        """搜索最相关的文档
        Args:
            tokenized_query: 已分词的查询词列表
            top_k: 返回的文档数量
        Returns:
            list: 包含相关文档的列表，每个文档是一个dict
        """
        if not self.bm25:
            return []
            
        # 获取文档得分
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top_k个最相关文档的索引
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            results.append({
                'score': float(doc_scores[idx]),
                'metadata': self.raw_documents[idx],
                'document': ' '.join(self.tokenized_documents[idx])
            })
            
        return results