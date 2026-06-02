"""
配置文件
集中管理项目的所有配置参数
"""

import os

# ==================== Ollama 配置 ====================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:7b")  # 或 qwen2:7b
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ==================== 文本切分配置 ====================
DEFAULT_CHUNK_SIZE = 1000  # 任务要求：chunk_size=1000
DEFAULT_CHUNK_OVERLAP = 200  # 任务要求：chunk_overlap=200

# ==================== 检索配置 ====================
DEFAULT_TOP_K = 3  # 任务要求：返回最相关的3个文本块

# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """你是一个专业的问答助手。请严格根据以下提供的参考文档内容来回答用户的问题。

重要规则：
1. 只能基于提供的参考文档内容回答
2. 如果文档中没有相关信息，必须明确说"文档中未找到相关答案"
3. 不要编造答案，不要添加文档中没有的信息
4. 回答要清晰、准确、简洁

参考文档内容：
{context}

用户问题：{question}

请用中文回答："""
