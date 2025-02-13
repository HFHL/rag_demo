import yaml
from .retriever import Retriever
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FaissRetriever(Retriever):
    def __init__(self, texts=None, raw_docs=None):
        """
        Initialize FAISS retriever
        Args:
            texts: List of preprocessed text documents
            raw_docs: Original documents for return
        """
        # 读取配置
        with open('/home/hhl/rag_test/rag_demo/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 获取当前使用的模型名称
        model_name = config['retriever']['active_model']
        
        # 初始化编码器
        self.encoder = SentenceTransformer(model_name)
        self.raw_docs = raw_docs
        self.index = None
        self.dimension = None
        
        # 如果提供了文本,则构建新索引
        if texts is not None:
            self._build_index(texts)

    def _build_index(self, texts):
        """构建FAISS索引"""
        # 编码文本
        text_vectors = []
        for doc in texts:
            if isinstance(doc, list):
                doc = ' '.join(doc)
            text_vectors.append(self.encoder.encode(doc))
            
        self.text_vectors = np.array(text_vectors).astype('float32')
        self.dimension = self.text_vectors.shape[1]
        
        # 构建索引
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.text_vectors)

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve most relevant documents for query
        Args:
            query: Query string
            top_k: Number of documents to return
        Returns:
            List of top k relevant documents
        """
        # 编码查询
        query_vector = self.encoder.encode(query)
        query_vector = np.array([query_vector]).astype('float32')
        
        # 搜索最相似的文档
        distances, indices = self.index.search(query_vector, top_k)
        
        # 返回原始文档
        results = []
        for idx in indices[0]:
            if isinstance(self.raw_docs[idx], dict):
                # 如果是字典格式,拼接title和text
                doc = self.raw_docs[idx]
                results.append(f"{doc['title']}\n{doc['text']}")
            else:
                results.append(self.raw_docs[idx])
        return results

