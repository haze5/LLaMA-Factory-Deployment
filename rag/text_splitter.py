"""
文本分块器模块
将长文档分割成小块
"""

from typing import List


class TextSplitter:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        初始化文本分块器

        Args:
            chunk_size: 每块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分割符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def split(self, text: str) -> List[str]:
        """
        分割文本

        Args:
            text: 待分割的文本

        Returns:
            分割后的文本块列表
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # 如果不是最后一块，尝试在分隔符处分割
            if end < len(text):
                best_split = end

                # 从后向前查找最佳分割点
                for sep in self.separators:
                    split_pos = text.rfind(sep, start, end)
                    if split_pos > start:
                        best_split = split_pos + len(sep)
                        break

                end = best_split

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 计算下一块的起始位置（考虑重叠）
            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        return chunks
