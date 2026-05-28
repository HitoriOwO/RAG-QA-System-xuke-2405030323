"""
RAG 核心流程模块
实现文档索引构建和检索增强生成
"""

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    VECTORSTORE_DIR,
)
from embeddings import get_embeddings
from document_loader import load_document, split_documents


class RAGPipeline:
    """RAG 问答流水线"""

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        vectorstore_dir: str = VECTORSTORE_DIR,
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.vectorstore_dir = vectorstore_dir
        self.vectorstore = None
        self.qa_chain = None

        # 确保向量存储目录存在
        os.makedirs(self.vectorstore_dir, exist_ok=True)

    def _get_embeddings(self):
        """获取嵌入模型"""
        return get_embeddings(model=self.embedding_model, base_url=self.ollama_base_url)

    def _get_llm(self):
        """获取大语言模型"""
        llm = Ollama(
            model=self.llm_model,
            base_url=self.ollama_base_url,
            temperature=0.3,
        )
        return llm

    def index_documents(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
        """
        加载文档并构建向量索引

        Args:
            file_path: 文档文件路径
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小

        Returns:
            索引的文本块数量
        """
        # 加载文档
        documents = load_document(file_path)
        print(f"已加载文档: {file_path}，共 {len(documents)} 页/段")

        # 切分文档
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"文档切分为 {len(chunks)} 个文本块")

        # 生成向量并存储
        embeddings = self._get_embeddings()

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        # 持久化向量存储
        self.vectorstore.save_local(self.vectorstore_dir)
        print(f"向量索引已保存到: {self.vectorstore_dir}")

        return len(chunks)

    def load_vectorstore(self) -> bool:
        """
        从磁盘加载已有的向量存储

        Returns:
            是否成功加载
        """
        index_path = os.path.join(self.vectorstore_dir, "index.faiss")
        if not os.path.exists(index_path):
            print("未找到已有的向量索引")
            return False

        try:
            embeddings = self._get_embeddings()
            self.vectorstore = FAISS.load_local(
                self.vectorstore_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print("向量索引加载成功")
            return True
        except Exception as e:
            print(f"加载向量索引失败: {e}")
            return False

    def _build_qa_chain(self, k: int = 4):
        """
        构建 RAG 问答链

        Args:
            k: 检索的文档数量
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化，请先索引文档或加载已有索引")

        llm = self._get_llm()

        # 自定义提示模板
        prompt_template = """你是一个专业的问答助手。请根据以下提供的参考文档内容来回答用户的问题。

如果你在参考文档中找不到答案，请诚实地告诉用户你不知道，不要编造答案。

参考文档内容：
{context}

用户问题：{question}

请用清晰、准确的语言回答："""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def query(self, question: str, k: int = 4) -> dict:
        """
        对文档进行问答

        Args:
            question: 用户问题
            k: 检索的文档数量

        Returns:
            包含答案和来源文档的字典
        """
        if self.qa_chain is None:
            self._build_qa_chain(k=k)

        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", []),
        }

    def get_index_info(self) -> Optional[dict]:
        """
        获取当前索引信息

        Returns:
            索引信息字典，如果没有索引则返回 None
        """
        index_path = os.path.join(self.vectorstore_dir, "index.faiss")
        if not os.path.exists(index_path):
            return None

        try:
            embeddings = self._get_embeddings()
            vs = FAISS.load_local(
                self.vectorstore_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            doc_count = vs.index.ntotal
            return {
                "indexed": True,
                "doc_count": doc_count,
                "index_path": self.vectorstore_dir,
            }
        except Exception:
            return None
