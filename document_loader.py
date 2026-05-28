"""
文档加载与文本切分模块
支持 PDF、TXT、Markdown 等格式的文档加载
"""

import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)


def load_document(file_path: str) -> List:
    """
    根据文件扩展名自动选择合适的加载器加载文档

    Args:
        file_path: 文档文件路径

    Returns:
        LangChain Document 对象列表
    """
    ext = os.path.splitext(file_path)[1].lower()

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
    }

    loader_class = loaders.get(ext)
    if loader_class is None:
        raise ValueError(f"不支持的文件格式: {ext}，支持的格式: {list(loaders.keys())}")

    loader = loader_class(file_path)
    documents = loader.load()
    return documents


def split_documents(documents: List, chunk_size: int = 500, chunk_overlap: int = 50) -> List:
    """
    将文档切分为较小的文本块

    Args:
        documents: LangChain Document 对象列表
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 文本块之间的重叠字符数

    Returns:
        切分后的 Document 对象列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
