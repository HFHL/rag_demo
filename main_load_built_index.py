import os
import pickle
import faiss
from retriever.rank_bm25_retriever import RankBM25Retriever
from retriever.faiss_retriever import FaissRetriever
from retriever.bm25s_retriever import BM25SRetriever

class IndexLoader:
    def __init__(self, index_dir="/home/hhl/rag_test/rag_demo/indexes"):
        """从保存的索引文件加载检索器"""
        self.index_dir = index_dir
        self.bm25_retriever = None
        self.faiss_retriever = None
        self.bm25s_retriever = None
        self.rank_bm25_retriever = RankBM25Retriever()
        
    def load_bm25_index(self):
        """加载BM25索引"""
        bm25_path = os.path.join(self.index_dir, "bm25.pkl")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
            
        print(f"Loading BM25 index from {bm25_path}")
        self.bm25_retriever = RankBM25Retriever.load(bm25_path)
    
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
    
    def load_bm25s_index(self):
        """加载BM25S索引"""
        bm25s_path = os.path.join(self.index_dir, "bm25s")
        if not os.path.exists(bm25s_path):
            raise FileNotFoundError(f"BM25S index not found at {bm25s_path}")
            
        print(f"Loading BM25S index from {bm25s_path}")
        # 创建空的检索器实例
        self.bm25s_retriever = BM25SRetriever()
        # 加载保存的索引
        self.bm25s_retriever.load(bm25s_path)
    
    def load_all_indexes(self):
        """加载所有索引"""
        self.load_bm25_index()
        self.load_bm25s_index()
        # self.load_faiss_index()

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
        tokenized_query = query.lower().split()  # 对查询进行分词
        bm25_results = loader.bm25_retriever.search(tokenized_query, top_k=top_k)
        for i, result in enumerate(bm25_results, 1):
            if isinstance(result, dict):
                doc_text = result.get('document', str(result))
            else:
                doc_text = str(result)
            preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            print(f"\n{i}. {preview}")

        print("\nBM25s Results:")
        # BM25S检索结果
        bm25s_results = loader.bm25s_retriever.search(query, top_k=top_k)
        for i, result in enumerate(bm25s_results, 1):
            if isinstance(result, dict):
                doc_text = result.get('document', str(result))
            else:
                doc_text = str(result)
            preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            print(f"\n{i}. {preview}")

        # RankBM25检索结果
        print("\nRankBM25 Results:")
        tokenized_query = query.lower().split()  # 对查询进行分词
        rank_bm25_results = loader.rank_bm25_retriever.search(tokenized_query, top_k=top_k)  # 改用search方法
        for i, result in enumerate(rank_bm25_results, 1):
            if isinstance(result, dict):
                doc_text = result.get('document', str(result))
            else:
                doc_text = str(result)
            preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            print(f"\n{i}. {preview}")
        
            
        # FAISS检索结果
        print("\nFAISS Results:")
        # faiss_results = loader.faiss_retriever.retrieve(query, top_k=top_k)
        # for i, doc in enumerate(faiss_results, 1):
        #     # 确保doc是字符串类型
        #     doc_text = str(doc)
        #     # 限制输出长度
        #     preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
        #     print(f"\n{i}. {preview}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run build_index.py first to create the index files.")

if __name__ == "__main__":
    main()
