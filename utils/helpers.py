"""
工具函数模块
提供项目中通用的辅助函数
"""

import os


def ensure_directory(path: str) -> str:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        目录路径
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名（小写，不含点号）

    Args:
        file_path: 文件路径

    Returns:
        文件扩展名
    """
    return os.path.splitext(file_path)[1].lower().lstrip(".")


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小为人类可读格式

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        格式化后的文件大小字符串
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 300, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix
