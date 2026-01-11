"""
向量数据库模块
使用 ChromaDB 存储和检索向量
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings


class VectorStore:
    """向量数据库封装"""

    def __init__(
        self,
        persist_dir: str,
        embedding_model,
        collection_name: str = "default"
    ):
        """
        初始化向量数据库

        Args:
            persist_dir: 持久化目录
            embedding_model: 嵌入模型
            collection_name: 集合名称
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        # 创建持久化目录
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 获取或创建集合
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """获取或创建集合"""
        try:
            # 尝试获取现有集合
            collection = self.client.get_collection(name=self.collection_name)
            return collection
        except Exception:
            # 集合不存在，创建新集合
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG knowledge base"}
            )

    def is_initialized(self) -> bool:
        """检查向量数据库是否已初始化"""
        try:
            count = self.collection.count()
            return count > 0
        except Exception:
            return False

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] = None
    ):
        """
        添加文档到向量数据库

        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表
            metadatas: 元数据列表
        """
        n_docs = len(texts)

        # 生成唯一 ID
        ids = [f"doc_{i}_{hash(texts[i]) % 1000000}" for i in range(n_docs)]

        # 如果没有元数据，创建空元数据
        if metadatas is None:
            metadatas = [{}] * n_docs

        # 批量添加文档
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        搜索相关文档

        Args:
            query_embedding: 查询向量
            top_k: 返回的结果数量
            where: 元数据过滤条件

        Returns:
            相关文档列表
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        # 提取文档内容
        documents = results.get('documents', [])
        if documents and len(documents) > 0:
            return documents[0]

        return []

    def reset(self):
        """清空向量数据库"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)

    def persist(self):
        """持久化向量数据库"""
        # ChromaDB 的 PersistentClient 会自动持久化
        pass
