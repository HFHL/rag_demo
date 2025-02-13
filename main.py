import sys
import json
import re
from retriever.rank_bm25_retriever import RankBM25Retriever
from retriever.faiss_retriever import FaissRetriever

def load_data(json_path):
    """Load and preprocess JSON data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    raw_documents = []
    
    for doc in data:
        # 只使用title和text字段
        title = doc['title']
        # 展平text列表并连接所有句子
        text_content = ' '.join([' '.join(p) for p in doc['text']])
        
        # 移除HTML标签
        text_content = re.sub(r'<[^>]+>', '', text_content)
        
        # 组合title和text,给title更高权重(重复两次)
        full_text = f"{title} {title} {text_content}"
        
        # Preprocess
        full_text = full_text.lower()
        full_text = re.sub(r'[^a-zA-Z0-9\s]', '', full_text)
        
        # Store processed and raw text
        documents.append(full_text.split())
        raw_documents.append(full_text)
    
    return documents, raw_documents

def main():
    # Load and preprocess data
    corpus, raw_docs = load_data('/home/hhl/rag_test/rag_demo/data/decompressed_files/wiki_01.json')
    
    # Initialize retrievers 
    bm25_retriever = RankBM25Retriever(corpus, raw_docs)
    faiss_retriever = FaissRetriever(corpus, raw_docs)
    
    # Test query
    query = "What is National Australia Bank?"
    top_k = 3
    
    # Get results from both retrievers
    print(f"Query: {query}\n")
    
    print("BM25 Results:")
    bm25_results = bm25_retriever.retrieve(query, top_k=top_k)
    for i, doc in enumerate(bm25_results, 1):
        print(f"\n{i}. {doc[:200]}...")
        
    print("\nFAISS Results:")  
    faiss_results = faiss_retriever.retrieve(query, top_k=top_k)
    for i, doc in enumerate(faiss_results, 1):
        print(f"\n{i}. {doc[:200]}...")

if __name__ == "__main__":
    main()