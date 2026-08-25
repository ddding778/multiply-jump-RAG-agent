"""RAG V2 Markdown 切分的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Heading:
    """表示一个 Markdown 标题节点。

    参数 level 为标题等级，text 为去除井号后的标题内容，line 为标题所在的源文件行号。
    """

    level: int
    text: str
    line: int

    def as_dict(self) -> dict[str, int | str]:
        """将标题转换为可写入 JSON 的数据。

        返回包含 level、text、line 的字典。
        """

        return {"level": self.level, "text": self.text, "line": self.line}


@dataclass
class MarkdownChunk:
    """表示一个可独立 embedding 的 Markdown 叶子 chunk。

    参数保存文档、section、标题路径、源行号、相邻节点和最终 embedding 文本；
    返回对象可通过 as_dict 序列化为索引 manifest 的一行。
    """

    document_id: str
    section_id: str
    parent_section_id: str | None
    chunk_id: str
    chunk_order: int
    heading_path: list[Heading]
    text: str
    embedding_text: str
    line_start: int
    line_end: int
    content_hash: str
    has_code: bool
    token_count: int
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    _section_key: str = field(default="", repr=False)

    def as_dict(self) -> dict[str, object]:
        """将 chunk 转换为后续 manifest 和 Qdrant payload 可使用的字典。

        无参数；返回不包含内部 section 排序键的公开元数据。
        """

        return {
            "document_id": self.document_id,
            "section_id": self.section_id,
            "parent_section_id": self.parent_section_id,
            "chunk_id": self.chunk_id,
            "chunk_order": self.chunk_order,
            "heading_path": [heading.as_dict() for heading in self.heading_path],
            "text": self.text,
            "embedding_text": self.embedding_text,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_hash": self.content_hash,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "has_code": self.has_code,
            "token_count": self.token_count,
        }
