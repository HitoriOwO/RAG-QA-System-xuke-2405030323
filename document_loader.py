"""
文档加载与文本切分模块
支持 PDF、TXT、Markdown、DOCX 等格式的文档加载
实现批量读取指定文件夹内所有文档
"""

import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def load_document(file_path: str) -> List[Document]:
    """
    根据文件扩展名自动选择合适的加载器加载单个文档

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


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    批量加载指定文件夹内的所有支持格式的文档
    任务要求：实现对指定文件夹内所有文档的批量读取与文本提取

    Args:
        directory_path: 文档文件夹路径

    Returns:
        LangChain Document 对象列表
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"目录不存在: {directory_path}")

    all_documents = []
    supported_extensions = [".pdf", ".txt", ".md", ".docx"]

    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext in supported_extensions:
                try:
                    docs = load_document(file_path)
                    all_documents.extend(docs)
                    print(f"✓ 已加载: {file}")
                except Exception as e:
                    print(f"✗ 加载失败 {file}: {e}")

    print(f"\n总计加载 {len(all_documents)} 个文档片段")
    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    将文档切分为较小的文本块
    任务要求：使用 RecursiveCharacterTextSplitter，chunk_size=1000，chunk_overlap=200

    Args:
        documents: LangChain Document 对象列表
        chunk_size: 每个文本块的最大字符数（默认1000）
        chunk_overlap: 文本块之间的重叠字符数（默认200）

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
    print(f"文档切分完成: {len(documents)} 个文档 → {len(chunks)} 个文本块")
    return chunks


def get_document_info(directory_path: str) -> dict:
    """
    获取指定目录中文档的信息统计

    Args:
        directory_path: 文档目录路径

    Returns:
        包含文档统计信息的字典
    """
    if not os.path.exists(directory_path):
        return {"error": "目录不存在"}

    file_counts = {"pdf": 0, "txt": 0, "md": 0, "docx": 0, "total": 0}
    total_size = 0

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            file_path = os.path.join(root, file)

            if ext in [".pdf", ".txt", ".md", ".docx"]:
                file_counts["total"] += 1
                file_counts[ext.lstrip(".")] += 1
                total_size += os.path.getsize(file_path)

    return {
        "file_counts": file_counts,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "directory": directory_path,
    }
