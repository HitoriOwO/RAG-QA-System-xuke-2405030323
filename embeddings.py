"""
向量嵌入模块
使用 Ollama 本地模型生成文本向量
支持 nomic-embed-text 和 all-minilm 等嵌入模型
"""

from langchain_community.embeddings import OllamaEmbeddings
from config import DEFAULT_EMBEDDING_MODEL, OLLAMA_BASE_URL


def get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
    """
    获取 Ollama 嵌入模型实例
    任务要求：使用 LangChain 的嵌入模型（nomic-embed-text 或 all-minilm）

    Args:
        model: 嵌入模型名称，可选：
               - nomic-embed-text: 高质量通用嵌入模型
               - all-minilm: 轻量级多语言嵌入模型
        base_url: Ollama 服务地址

    Returns:
        OllamaEmbeddings 实例
    """
    embeddings = OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )
    return embeddings


def get_available_embedding_models() -> list:
    """
    获取推荐的嵌入模型列表

    Returns:
        推荐模型名称列表
    """
    return [
        "nomic-embed-text",  # 推荐：高质量、768维
        "all-minilm",        # 轻量级、384维、多语言
        "mxbai-embed-large", # 高性能、1024维
    ]
