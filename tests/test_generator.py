import unittest
from generator import rag_generator

class TestRAGGenerator(unittest.TestCase):
    def test_generate(self):
        generator = rag_generator.RAGGenerator()
        result = generator.generate("问题", ["相关文档1", "相关文档2"])
        self.assertEqual(result, "生成的答案")

if __name__ == '__main__':
    unittest.main()
