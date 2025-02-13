# retriever/retriever.py

from abc import ABC, abstractmethod

class Retriever(ABC):
    """
    抽象检索器基类，所有的检索器都需要继承并实现 `retrieve` 方法
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5):
        """
        通过查询从数据库中检索相关文档

        :param query: 用户的查询字符串
        :param top_k: 返回的相关文档数
        :return: 返回相关文档列表
        """
        pass