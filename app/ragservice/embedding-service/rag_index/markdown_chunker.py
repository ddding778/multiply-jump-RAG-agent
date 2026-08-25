"""基于 Markdown 结构和 token 预算生成可检索叶子 chunk。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import Heading, MarkdownChunk


_ATX_HEADING = re.compile(r"^(?: {0,3})(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$")
_FENCE_START = re.compile(r"^(?: {0,3})(`{3,}|~{3,})(.*)$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])")


class TokenCounter(Protocol):
    """定义 token 计数器合同。

    count 接收文本并返回该文本的 token 数；实际 embedding 模型更换时可替换实现。
    """

    def count(self, text: str) -> int:
        """计算文本 token 数。

        参数 text 为待计算文本；返回非负 token 数。
        """


class TiktokenTokenCounter:
    """使用 tiktoken 的指定 encoding 估算 token 数。

    参数 encoding_name 为 tiktoken encoding 名称；返回实例可传给 MarkdownChunker。
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """初始化指定 encoding 的计数器。

        参数 encoding_name 为 tiktoken 已注册的编码名称；无返回值。
        """

        try:
            import tiktoken
        except ImportError as error:  # pragma: no cover - 依赖缺失由运行环境决定。
            raise RuntimeError("缺少 tiktoken，请先安装 requirements.txt 中的依赖") from error
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """估算文本的 token 数。

        参数 text 为待计算文本；返回 tiktoken 编码后的 token 数。
        """

        return len(self._encoding.encode(text, disallowed_special=()))


@dataclass
class _Block:
    """保存 section 内尚未切分的自然语义块。

    参数 text 为原始文本，line_start 与 line_end 为源行号，kind 为 prose、code、list、table 或 quote。
    """

    text: str
    line_start: int
    line_end: int
    kind: str


@dataclass
class _Section:
    """保存一个标题区间及其自然语义块。

    参数 heading_path 为当前完整标题栈，parent_section_id 为直接父标题节点；返回对象只用于内部切分。
    """

    heading_path: list[Heading]
    section_id: str
    parent_section_id: str | None
    blocks: list[_Block]


class MarkdownChunker:
    """以标题树、自然块和双 token 阈值切分 Markdown 文档。

    参数 token_counter 用于计数；soft_limit_tokens 为自然边界优先落块目标，
    hard_limit_tokens 为单个 embedding 文本绝不跨越的硬上限。返回 chunk_markdown 生成的叶子 chunk 列表。
    """

    def __init__(self, token_counter: TokenCounter, soft_limit_tokens: int = 800, hard_limit_tokens: int = 1200) -> None:
        """初始化切分器并校验 token 阈值。

        参数 token_counter 为实际计数器，soft_limit_tokens 为优先切分阈值，hard_limit_tokens 为硬阈值；
        阈值非法时抛出 ValueError。
        """

        if soft_limit_tokens <= 0:
            raise ValueError("soft_limit_tokens must be positive")
        if hard_limit_tokens < soft_limit_tokens:
            raise ValueError("hard_limit_tokens must be greater than or equal to soft_limit_tokens")
        self._token_counter = token_counter
        self._soft_limit_tokens = soft_limit_tokens
        self._hard_limit_tokens = hard_limit_tokens

    def chunk_file(self, file_path: Path, docs_root: Path) -> list[MarkdownChunk]:
        """读取 docs 根目录内的单个 Markdown 文件并完成切分。

        参数 file_path 为待切分文件，docs_root 为允许的文档根目录；返回按文档顺序排列的 chunk 列表。
        """

        relative_path = file_path.resolve().relative_to(docs_root.resolve()).as_posix()
        return self.chunk_markdown(file_path.read_text(encoding="utf-8"), relative_path)

    def chunk_markdown(self, markdown: str, relative_path: str) -> list[MarkdownChunk]:
        """切分一份 Markdown 文本。

        参数 markdown 为 UTF-8 文本内容，relative_path 为 docs 根目录下的 POSIX 相对路径；
        返回只包含可 embedding 叶子节点的 chunk 列表。
        """

        document_id = _stable_id("document", relative_path)
        sections = self._parse_sections(markdown, document_id)
        chunks: list[MarkdownChunk] = []
        for section in sections:
            chunks.extend(self._chunk_section(section, document_id, len(chunks)))
        self._link_neighbors(chunks)
        return chunks

    def _parse_sections(self, markdown: str, document_id: str) -> list[_Section]:
        """按标题栈和 fenced-code 状态将文档拆为标题区间。

        参数 markdown 为原始 Markdown，document_id 为稳定文档标识；返回含自然语义块的 section 列表。
        """

        lines = markdown.splitlines()
        heading_stack: list[Heading] = []
        sections: list[_Section] = []
        pending_lines: list[tuple[int, str]] = []
        in_fence: tuple[str, int] | None = None

        def flush_section() -> None:
            """将当前标题区间的缓冲行转换为 section。

            无参数；返回值为空，纯追加到外层 sections。
            """

            nonlocal pending_lines
            if not pending_lines:
                return
            section_id = _section_id(document_id, heading_stack)
            parent_section_id = _section_id(document_id, heading_stack[:-1]) if heading_stack else None
            blocks = _build_blocks(pending_lines)
            if blocks:
                sections.append(
                    _Section(
                        heading_path=list(heading_stack),
                        section_id=section_id,
                        parent_section_id=parent_section_id,
                        blocks=blocks,
                    )
                )
            pending_lines = []

        for line_number, line in enumerate(lines, start=1):
            if in_fence is not None:
                pending_lines.append((line_number, line))
                if _is_fence_close(line, in_fence[0], in_fence[1]):
                    in_fence = None
                continue

            fence = _FENCE_START.match(line)
            if fence:
                pending_lines.append((line_number, line))
                in_fence = (fence.group(1)[0], len(fence.group(1)))
                continue

            heading_match = _ATX_HEADING.match(line)
            if heading_match:
                flush_section()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                while heading_stack and heading_stack[-1].level >= level:
                    heading_stack.pop()
                heading_stack.append(Heading(level=level, text=title, line=line_number))
                continue

            pending_lines.append((line_number, line))

        flush_section()
        return sections

    def _chunk_section(self, section: _Section, document_id: str, start_chunk_order: int) -> list[MarkdownChunk]:
        """在一个标题区间内按自然块和 token 阈值生成叶子 chunk。

        参数 section 为已解析标题区间，document_id 为稳定文档标识，start_chunk_order 为文档内起始顺序；
        返回该 section 的有序 chunk 列表，chunk_order 在整份文档内连续递增。
        """

        parts: list[_Block] = []
        for block in section.blocks:
            parts.extend(self._split_oversized_block(block, section.heading_path))

        chunks: list[MarkdownChunk] = []
        current_parts: list[_Block] = []
        for part in parts:
            candidate_parts = [*current_parts, part]
            candidate_text = _join_parts(candidate_parts)
            if current_parts and self._embedding_token_count(section.heading_path, candidate_text) > self._hard_limit_tokens:
                chunks.append(self._make_chunk(section, document_id, start_chunk_order + len(chunks), current_parts))
                current_parts = [part]
            else:
                current_parts = candidate_parts

            if self._embedding_token_count(section.heading_path, _join_parts(current_parts)) >= self._soft_limit_tokens:
                chunks.append(self._make_chunk(section, document_id, start_chunk_order + len(chunks), current_parts))
                current_parts = []

        if current_parts:
            chunks.append(self._make_chunk(section, document_id, start_chunk_order + len(chunks), current_parts))
        return chunks

    def _split_oversized_block(self, block: _Block, heading_path: list[Heading]) -> list[_Block]:
        """只在单个自然块超过硬上限时，在块内部寻找安全边界。

        参数 block 为自然块，heading_path 为重复到每个子块的标题路径；返回不超过硬上限的子块列表。
        """

        if self._embedding_token_count(heading_path, block.text) <= self._hard_limit_tokens:
            return [block]
        if block.kind in {"prose", "quote"}:
            return self._split_prose(block, heading_path)
        return self._split_by_line(block, heading_path)

    def _split_prose(self, block: _Block, heading_path: list[Heading]) -> list[_Block]:
        """优先在句末切分过长正文或引用块。

        参数 block 为超长正文块，heading_path 为标题路径；返回在句号等边界切开的子块，无法满足硬上限时再降级。
        """

        sentences = _sentence_blocks(block)
        if not sentences:
            return self._force_split(block, heading_path)

        result: list[_Block] = []
        current: list[_Block] = []
        for sentence in sentences:
            candidate = [*current, sentence]
            candidate_text = _join_parts(candidate)
            if current and self._embedding_token_count(heading_path, candidate_text) > self._hard_limit_tokens:
                result.append(_merge_blocks(current, "prose"))
                current = [sentence]
            else:
                current = candidate

            if self._embedding_token_count(heading_path, _join_parts(current)) >= self._soft_limit_tokens:
                result.append(_merge_blocks(current, "prose"))
                current = []

        if current:
            result.append(_merge_blocks(current, "prose"))

        safe_result: list[_Block] = []
        for item in result:
            if self._embedding_token_count(heading_path, item.text) <= self._hard_limit_tokens:
                safe_result.append(item)
            else:
                safe_result.extend(self._force_split(item, heading_path))
        return safe_result

    def _split_by_line(self, block: _Block, heading_path: list[Heading]) -> list[_Block]:
        """在代码、列表或表格超过硬上限时按完整行切分。

        参数 block 为超长结构块，heading_path 为标题路径；返回尽量保留原始行边界的子块。
        """

        lines = block.text.splitlines(keepends=True)
        result: list[_Block] = []
        current: list[_Block] = []
        line_number = block.line_start
        for raw_line in lines:
            line = _Block(raw_line.rstrip("\n"), line_number, line_number, block.kind)
            line_number += raw_line.count("\n")
            candidate = [*current, line]
            if current and self._embedding_token_count(heading_path, _join_parts(candidate)) > self._hard_limit_tokens:
                result.append(_merge_blocks(current, block.kind))
                current = [line]
            else:
                current = candidate
        if current:
            result.append(_merge_blocks(current, block.kind))

        safe_result: list[_Block] = []
        for item in result:
            if self._embedding_token_count(heading_path, item.text) <= self._hard_limit_tokens:
                safe_result.append(item)
            else:
                safe_result.extend(self._force_split(item, heading_path))
        return safe_result

    def _force_split(self, block: _Block, heading_path: list[Heading]) -> list[_Block]:
        """处理单句或单行已超过硬上限的不可避免情形。

        参数 block 为无法在自然边界容纳的文本，heading_path 为标题路径；返回严格受硬上限约束的子块。
        """

        if self._embedding_token_count(heading_path, "") >= self._hard_limit_tokens:
            raise ValueError("标题路径本身已达到或超过 hard_limit_tokens")

        remaining = block.text
        result: list[_Block] = []
        offset = 0
        while remaining:
            if self._embedding_token_count(heading_path, remaining) <= self._hard_limit_tokens:
                result.append(
                    _Block(
                        remaining,
                        block.line_start + block.text[:offset].count("\n"),
                        block.line_end,
                        block.kind,
                    )
                )
                break

            split_at = self._largest_safe_prefix(remaining, heading_path)
            boundary = _preferred_boundary(remaining, split_at)
            if boundary <= 0:
                boundary = split_at
            piece = remaining[:boundary].rstrip()
            if not piece:
                piece = remaining[:split_at]
                boundary = split_at
            result.append(
                _Block(
                    piece,
                    block.line_start + block.text[:offset].count("\n"),
                    block.line_start + block.text[: offset + boundary].count("\n"),
                    block.kind,
                )
            )
            offset += boundary
            remaining = remaining[boundary:].lstrip()
        return result

    def _largest_safe_prefix(self, text: str, heading_path: list[Heading]) -> int:
        """查找在硬上限内的最长非空字符前缀。

        参数 text 为待切分文本，heading_path 为标题路径；返回可安全 embedding 的前缀长度。
        """

        lower, upper = 1, len(text)
        best = 0
        while lower <= upper:
            middle = (lower + upper) // 2
            if self._embedding_token_count(heading_path, text[:middle]) <= self._hard_limit_tokens:
                best = middle
                lower = middle + 1
            else:
                upper = middle - 1
        while best > 0 and self._embedding_token_count(heading_path, text[:best]) > self._hard_limit_tokens:
            best -= 1
        if best == 0:
            raise ValueError("hard_limit_tokens 不足以容纳最小文本单元")
        return best

    def _make_chunk(self, section: _Section, document_id: str, chunk_order: int, parts: list[_Block]) -> MarkdownChunk:
        """从已确定的自然块创建带稳定 ID 的叶子 chunk。

        参数 section 为所属标题区间，document_id 为文档标识，chunk_order 为 section 内顺序，parts 为正文块；
        返回完整的 MarkdownChunk。
        """

        text = _join_parts(parts)
        embedding_text = _embedding_text(section.heading_path, text)
        token_count = self._token_counter.count(embedding_text)
        if token_count > self._hard_limit_tokens:
            raise ValueError("chunk exceeds hard_limit_tokens")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk_id = _stable_id("chunk", document_id, section.section_id, str(chunk_order), content_hash)
        return MarkdownChunk(
            document_id=document_id,
            section_id=section.section_id,
            parent_section_id=section.parent_section_id,
            chunk_id=chunk_id,
            chunk_order=chunk_order,
            heading_path=list(section.heading_path),
            text=text,
            embedding_text=embedding_text,
            line_start=parts[0].line_start,
            line_end=parts[-1].line_end,
            content_hash=content_hash,
            has_code=any(part.kind == "code" for part in parts),
            token_count=token_count,
            _section_key=section.section_id,
        )

    def _embedding_token_count(self, heading_path: list[Heading], text: str) -> int:
        """计算带标题路径的最终 embedding 文本 token 数。

        参数 heading_path 为完整标题路径，text 为正文；返回 embedding 输入的 token 数。
        """

        return self._token_counter.count(_embedding_text(heading_path, text))

    def _link_neighbors(self, chunks: list[MarkdownChunk]) -> None:
        """只在同一 section 内连接前后叶子 chunk。

        参数 chunks 为整个文档的有序 chunk；无返回值，直接写入 previous_chunk_id 与 next_chunk_id。
        """

        by_section: dict[str, list[MarkdownChunk]] = {}
        for chunk in chunks:
            by_section.setdefault(chunk._section_key, []).append(chunk)
        for section_chunks in by_section.values():
            for index, chunk in enumerate(section_chunks):
                if index > 0:
                    chunk.previous_chunk_id = section_chunks[index - 1].chunk_id
                if index + 1 < len(section_chunks):
                    chunk.next_chunk_id = section_chunks[index + 1].chunk_id


def _build_blocks(lines: list[tuple[int, str]]) -> list[_Block]:
    """将标题区间的行按段落和 fenced code block 构造成自然块。

    参数 lines 为带源行号的行序列；返回保留列表、表格、引用和代码边界的块列表。
    """

    blocks: list[_Block] = []
    current: list[tuple[int, str]] = []
    fence: tuple[str, int] | None = None

    def flush_current() -> None:
        """将普通缓冲行提交为一个自然块。

        无参数；返回值为空，直接追加到外层 blocks。
        """

        nonlocal current
        if not current:
            return
        block_text = "\n".join(text for _, text in current).strip()
        if block_text and not _is_non_indexable_control_block(block_text):
            blocks.append(
                _Block(
                    text=block_text,
                    line_start=current[0][0],
                    line_end=current[-1][0],
                    kind=_block_kind([text for _, text in current]),
                )
            )
        current = []

    for line_number, line in lines:
        if fence is not None:
            current.append((line_number, line))
            if _is_fence_close(line, fence[0], fence[1]):
                flush_current()
                fence = None
            continue

        fence_match = _FENCE_START.match(line)
        if fence_match:
            flush_current()
            current.append((line_number, line))
            fence = (fence_match.group(1)[0], len(fence_match.group(1)))
            continue

        if not line.strip():
            flush_current()
            continue
        current.append((line_number, line))
    flush_current()
    return blocks


def _block_kind(lines: list[str]) -> str:
    """根据块的首行和内容判断自然块类型。

    参数 lines 为同一自然块的文本行；返回 prose、list、table、quote 或 code。
    """

    first = lines[0].lstrip()
    if _FENCE_START.match(lines[0]):
        return "code"
    if first.startswith(">"):
        return "quote"
    if re.match(r"(?:[-+*]|\d+[.)])\s+", first):
        return "list"
    if len(lines) >= 2 and "|" in lines[0] and re.match(r"^\s*\|?\s*:?-{3,}", lines[1]):
        return "table"
    return "prose"


def _is_non_indexable_control_block(text: str) -> bool:
    """判断文本是否为没有检索价值的独立 Markdown 控制标记。

    参数 text 为去除首尾空白后的自然块文本；返回是否应在切分阶段跳过。
    仅跳过明确列出的 TOC 标记，避免因正文过短而误删有效内容。
    """

    normalized = re.sub(r"\s+", "", text).casefold()
    return normalized in {
        "[toc]",
        "[[toc]]",
        "<!--toc-->",
        "<!--/toc-->",
        "<!--toc:start-->",
        "<!--toc:end-->",
    }


def _sentence_blocks(block: _Block) -> list[_Block]:
    """将正文块按中英文句末标点拆分，并保留近似源行号。

    参数 block 为 prose 或 quote 块；返回不丢弃原文的句子块列表。
    """

    pieces = [piece for piece in _SENTENCE_BOUNDARY.split(block.text) if piece]
    if len(pieces) <= 1:
        return [block]
    result: list[_Block] = []
    offset = 0
    for piece in pieces:
        start_line = block.line_start + block.text[:offset].count("\n")
        offset += len(piece)
        end_line = block.line_start + block.text[:offset].count("\n")
        result.append(_Block(piece.strip(), start_line, end_line, block.kind))
    return [item for item in result if item.text]


def _merge_blocks(blocks: list[_Block], kind: str) -> _Block:
    """合并相邻子块并保留其覆盖的源行区间。

    参数 blocks 为至少一个连续块，kind 为合并后的类型；返回合并结果。
    """

    return _Block(_join_parts(blocks), blocks[0].line_start, blocks[-1].line_end, kind)


def _join_parts(parts: list[_Block]) -> str:
    """以空行连接自然块。

    参数 parts 为有序自然块；返回保留块边界的正文文本。
    """

    return "\n\n".join(part.text for part in parts if part.text).strip()


def _embedding_text(heading_path: list[Heading], text: str) -> str:
    """生成实际发送给 embedding API 的带标题上下文文本。

    参数 heading_path 为完整标题栈，text 为 chunk 正文；返回可直接向量化的文本。
    """

    if not heading_path:
        return "正文：\n" + text
    path = " > ".join(heading.text for heading in heading_path)
    return f"标题路径：{path}\n\n正文：\n{text}"


def _section_id(document_id: str, heading_path: list[Heading]) -> str:
    """为标题路径生成稳定 section ID。

    参数 document_id 为文档 ID，heading_path 为完整标题栈；返回稳定哈希 ID。
    """

    if not heading_path:
        return _stable_id("section", document_id, "root")
    serialized_path = "/".join(f"{item.level}:{item.line}:{item.text}" for item in heading_path)
    return _stable_id("section", document_id, serialized_path)


def _stable_id(*parts: str) -> str:
    """根据稳定输入生成 SHA-256 标识。

    参数 parts 为有序字符串组成部分；返回十六进制 SHA-256 哈希。
    """

    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _is_fence_close(line: str, fence_char: str, minimum_length: int) -> bool:
    """判断一行是否关闭当前 fenced code block。

    参数 line 为当前行，fence_char 为围栏字符，minimum_length 为起始围栏长度；返回是否闭合。
    """

    pattern = rf"^(?: {{0,3}}){re.escape(fence_char)}{{{minimum_length},}}[ \t]*$"
    return re.match(pattern, line) is not None


def _preferred_boundary(text: str, maximum: int) -> int:
    """在安全前缀内优先寻找句末、空白或换行边界。

    参数 text 为剩余文本，maximum 为最长安全前缀长度；返回推荐切分位置，找不到时返回 maximum。
    """

    prefix = text[:maximum]
    for marker in ("。", "！", "？", ".", "!", "?", "；", ";", "\n", " "):
        position = prefix.rfind(marker)
        if position >= 0:
            return position + len(marker)
    return maximum
