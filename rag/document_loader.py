"""
文档加载器模块
支持多种格式的文档加载
"""

from pathlib import Path
from typing import List
import json


class DocumentLoader:
    """文档加载器"""

    def load(self, file_path: str) -> List[str]:
        """
        加载文档内容

        Args:
            file_path: 文件路径

        Returns:
            文档内容列表
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 根据文件扩展名选择加载方式
        suffix = path.suffix.lower()

        if suffix in ['.txt', '.md']:
            return self._load_text(path)
        elif suffix in ['.json']:
            return self._load_json(path)
        else:
            # 其他格式尝试作为文本处理
            return self._load_text(path)

    def _load_text(self, path: Path) -> List[str]:
        """加载纯文本文件"""
        try:
            content = path.read_text(encoding='utf-8')
            if content.strip():
                return [content]
            return []
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                content = path.read_text(encoding='gbk')
                if content.strip():
                    return [content]
                return []
            except Exception:
                return []

    def _load_json(self, path: Path) -> List[str]:
        """加载 JSON 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理不同的 JSON 结构
        if isinstance(data, list):
            return [json.dumps(item, ensure_ascii=False) for item in data]
        elif isinstance(data, dict):
            # 尝试提取对话内容
            if 'conversations' in data:
                texts = []
                for conv in data['conversations']:
                    if 'value' in conv:
                        texts.append(conv['value'])
                return texts
            return [json.dumps(data, ensure_ascii=False)]
        else:
            return [str(data)]
