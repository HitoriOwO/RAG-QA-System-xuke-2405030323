"""
向量嵌入模块
使用 Ollama 本地模型生成文本向量
"""

from langchain_community.embeddings import OllamaEmbeddings
from config import DEFAULT_EMBEDDING_MODEL, OLLAMA_BASE_URL


def get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
    """
    获取 Ollama 嵌入模型实例

    Args:
        model: 嵌入模型名称
        base_url: Ollama 服务地址

    Returns:
        OllamaEmbeddings 实例
    """
    embeddings = OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )
    return embeddings
