import os
import pickle
import faiss
from retriever.rank_bm25_retriever import RankBM25Retriever
from retriever.faiss_retriever import FaissRetriever

class IndexLoader:
    def __init__(self, index_dir="/home/hhl/rag_test/rag_demo/indexes"):
        """从保存的索引文件加载检索器"""
        self.index_dir = index_dir
        self.bm25_retriever = None
        self.faiss_retriever = None
        
    def load_bm25_index(self):
        """加载BM25索引"""
        bm25_path = os.path.join(self.index_dir, "bm25.pkl")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
            
        print(f"Loading BM25 index from {bm25_path}")
        with open(bm25_path, 'rb') as f:
            self.bm25_retriever = pickle.load(f)
    
    def load_faiss_index(self):
        """加载FAISS索引"""
        faiss_index_path = os.path.join(self.index_dir, "faiss.index")
        faiss_docs_path = os.path.join(self.index_dir, "faiss_docs.pkl")
        
        if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_docs_path):
            raise FileNotFoundError(f"FAISS index files not found in {self.index_dir}")
            
        print(f"Loading FAISS index from {self.index_dir}")
        
        # 加载索引
        index = faiss.read_index(faiss_index_path)
        
        # 加载文档数据
        with open(faiss_docs_path, 'rb') as f:
            data = pickle.load(f)
            
        # 创建检索器实例
        self.faiss_retriever = FaissRetriever(raw_docs=data["raw_docs"])  # 使用关键字参数
        self.faiss_retriever.index = index
        self.faiss_retriever.dimension = data["dimension"]
    
    def load_all_indexes(self):
        """加载所有索引"""
        self.load_bm25_index()
        self.load_faiss_index()

def main():
    # 初始化加载器
    loader = IndexLoader()
    
    try:
        # 加载所有索引
        loader.load_all_indexes()
        
        # 测试查询
        query = "What is National Australia Bank?"
        top_k = 5
        
        print(f"\nQuery: {query}")
        
        # BM25检索结果
        print("\nBM25 Results:")
        bm25_results = loader.bm25_retriever.retrieve(query, top_k=top_k)
        for i, doc in enumerate(bm25_results, 1):
            # 确保doc是字符串类型
            doc_text = str(doc)
            # 限制输出长度
            preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            print(f"\n{i}. {preview}")
            
        # FAISS检索结果
        print("\nFAISS Results:")
        faiss_results = loader.faiss_retriever.retrieve(query, top_k=top_k)
        for i, doc in enumerate(faiss_results, 1):
            # 确保doc是字符串类型
            doc_text = str(doc)
            # 限制输出长度
            preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            print(f"\n{i}. {preview}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run build_index.py first to create the index files.")

if __name__ == "__main__":
    main()
