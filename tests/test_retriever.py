import unittest
from retriever import bm25_retriever, faiss_retriever

class TestBM25Retriever(unittest.TestCase):
    def test_retrieve(self):
        retriever = bm25_retriever.BM25Retriever()
        result = retriever.retrieve("问题")
        self.assertEqual(result, ["相关文档1", "相关文档2"])

class TestFaissRetriever(unittest.TestCase):
    def test_retrieve(self):
        retriever = faiss_retriever.FaissRetriever()
        result = retriever.retrieve("问题")
        self.assertEqual(result, ["相关文档1", "相关文档2"])

if __name__ == '__main__':
    unittest.main()
