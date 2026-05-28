"""
配置文件
集中管理项目的所有配置参数
"""

import os

# ==================== Ollama 配置 ====================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# ==================== 文本切分配置 ====================
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# ==================== 检索配置 ====================
DEFAULT_TOP_K = 4
