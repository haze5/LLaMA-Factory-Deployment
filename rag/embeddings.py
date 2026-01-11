"""
嵌入模型模块
将文本转换为向量表示
"""

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """嵌入模型封装"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化嵌入模型

        Args:
            model_name: 模型名称或本地路径
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        """
        将查询文本转换为向量

        Args:
            text: 查询文本

        Returns:
            向量列表
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将文档列表转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()
