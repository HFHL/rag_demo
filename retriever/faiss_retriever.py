import yaml
from .retriever import Retriever
import numpy as np
import faiss
from tqdm import tqdm
import os
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

class FaissRetriever(Retriever):
    def __init__(self, texts=None, raw_docs=None, model_name="bert-base-uncased"):
        """初始化FAISS检索器
        Args:
            texts: 文档文本列表(可选)
            raw_docs: 原始文档列表(可选)
            model_name: 选择加载的HuggingFace模型（默认为BERT）
        """
        print("Initializing FaissRetriever...")
        
        # 默认使用GPU
        self.use_gpu = True  # 默认启用GPU
        self.gpu_id = 0  # 使用GPU 0
        
        # 初始化编码器和其他属性 
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Load success
        print("Model loaded successfully.")
        self.raw_docs = raw_docs
        self.index = None
        self.dimension = None
        
        # 如果提供了文本，则构建新索引
        if texts is not None:
            print("Building FAISS index...")
            self._build_index(texts)

    def _to_gpu(self, index):
        """将索引转移到GPU"""
        print("Moving index to GPU...")
        try:
            # 尝试导入GPU版本的faiss
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
            return gpu_index
        except (ImportError, AttributeError):
            print("Warning: GPU version of FAISS not available, falling back to CPU")
            return index

    def _encode_batch(self, batch):
        """通过模型编码文本"""
        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # 获取句子级别的嵌入
        return embeddings

    def _build_index(self, texts):
        """构建FAISS索引，使用GPU加速"""
        print("Encoding documents with transformer...")
        text_vectors = []
        
        # 分批处理以节省内存
        batch_size = 64
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch = texts[i:i + batch_size]
            # 确保batch中的文本都是字符串
            batch = [' '.join(doc) if isinstance(doc, list) else doc for doc in batch]
            # 批量编码
            vectors = self._encode_batch(batch)
            text_vectors.extend(vectors)
        
        print("Converting to numpy array...")
        self.text_vectors = np.array(text_vectors).astype('float32')
        self.dimension = self.text_vectors.shape[1]
        
        print(f"Building FAISS index with dimension {self.dimension}...")
        
        # 构建Faiss索引，使用GPU加速
        res = faiss.StandardGpuResources()  # 创建GPU资源
        cpu_index = faiss.IndexFlatL2(self.dimension)  # 创建一个L2距离的索引
        self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)  # 将索引移到GPU
        
        print("Adding vectors to FAISS index...")
        # 分批添加向量
        batch_size = 1000
        total_batches = (len(self.text_vectors) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(self.text_vectors), batch_size), 
                     desc="Adding vectors to FAISS index",
                     total=total_batches):
            batch = self.text_vectors[i:i + batch_size]
            self.index.add(batch)  # 批量添加向量到索引
        
        print(f"FAISS index built successfully with {self.index.ntotal} vectors")

    def save(self, path: str):
        """保存检索器到文件"""
        print("Saving FAISS index and data...")
        # 创建保存目录
        os.makedirs(path, exist_ok=True)
        
        # 保存FAISS索引
        print("Saving FAISS index...")
        try:
            # 检查是否是GPU索引
            if hasattr(self.index, 'getDevice'):  # GPU index check
                print("Converting GPU index to CPU for saving...")
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
                
            # 保存CPU索引
            faiss.write_index(cpu_index, os.path.join(path, "faiss.index"))
            print("FAISS index saved successfully")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            raise  # Re-raise the exception for proper error handling
        
        # 保存其他数据
        with open(os.path.join(path, "retriever_data.pkl"), 'wb') as f:
            pickle.dump({
                'raw_docs': self.raw_docs,
                'dimension': self.dimension
            }, f)

    def load(self, path: str):
        """从文件加载检索器"""
        print("Loading FAISS index and data...")
        # 加载FAISS索引
        try:
            index = faiss.read_index(os.path.join(path, "faiss.index"))
            if self.use_gpu:
                # 如果启用GPU，将索引移至GPU
                self.index = self._to_gpu(index)
            else:
                self.index = index
            print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
        
        # 加载其他数据
        with open(os.path.join(path, "retriever_data.pkl"), 'rb') as f:
            data = pickle.load(f)
            self.raw_docs = data['raw_docs']
            self.dimension = data['dimension']

    def retrieve(self, query: str, top_k: int = 5):
        """检索相关文档"""
        print("Retrieving documents for query...")
        # 编码查询
        query_vector = self._encode_batch([query])
        query_vector = np.array([query_vector[0]]).astype('float32')
        
        # 搜索最相似的文档
        distances, indices = self.index.search(query_vector, top_k)
        
        print(f"Found {len(indices[0])} relevant documents.")
        
        # 返回原始文档
        results = []
        for idx in indices[0]:
            doc = self.raw_docs[idx]
            if doc.get('type') == 'title':
                result = f"Title: {doc['text']}"
            elif doc.get('type') == 'paragraph':
                result = f"Title: {doc['title']}\nContent: {doc['text']}"
            else:
                result = str(doc)
            results.append(result)
            
        return results