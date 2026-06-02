"""
RAG 核心流程模块
实现文档索引构建和检索增强生成
使用 Chroma 向量数据库和 ConversationalRetrievalChain
"""

import os
from typing import List, Optional, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    CHROMA_DB_DIR,
    DEFAULT_TOP_K,
    SYSTEM_PROMPT,
)
from embeddings import get_embeddings
from document_loader import load_documents_from_directory, split_documents


class RAGPipeline:
    """RAG 问答流水线 - 使用 ConversationalRetrievalChain 实现对话记忆"""

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        chroma_db_dir: str = CHROMA_DB_DIR,
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.chroma_db_dir = chroma_db_dir
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None

        # 确保向量存储目录存在
        os.makedirs(self.chroma_db_dir, exist_ok=True)

    def _get_embeddings(self):
        """获取嵌入模型"""
        return get_embeddings(model=self.embedding_model, base_url=self.ollama_base_url)

    def _get_llm(self):
        """获取大语言模型"""
        llm = ChatOllama(
            model=self.llm_model,
            base_url=self.ollama_base_url,
            temperature=0.3,
        )
        return llm

    def index_documents_from_directory(
        self,
        directory_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> int:
        """
        从指定文件夹批量加载文档并构建向量索引
        任务要求：实现对指定文件夹内所有文档的批量读取与文本提取

        Args:
            directory_path: 文档文件夹路径
            chunk_size: 文本块大小（默认1000）
            chunk_overlap: 文本块重叠大小（默认200）

        Returns:
            索引的文本块数量
        """
        # 批量加载文档
        documents = load_documents_from_directory(directory_path)
        if not documents:
            print("未找到任何文档")
            return 0

        # 切分文档
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"文档切分为 {len(chunks)} 个文本块")

        # 生成向量并存储到 Chroma
        embeddings = self._get_embeddings()

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.chroma_db_dir,
            )
        else:
            self.vectorstore.add_documents(chunks)

        # 持久化
        self.vectorstore.persist()
        print(f"向量索引已保存到 Chroma: {self.chroma_db_dir}")

        return len(chunks)

    def index_single_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> int:
        """
        加载单个文档并构建向量索引

        Args:
            file_path: 文档文件路径
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小

        Returns:
            索引的文本块数量
        """
        from document_loader import load_document, split_documents

        # 加载文档
        documents = load_document(file_path)
        print(f"已加载文档: {file_path}，共 {len(documents)} 页/段")

        # 切分文档
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"文档切分为 {len(chunks)} 个文本块")

        # 生成向量并存储到 Chroma
        embeddings = self._get_embeddings()

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.chroma_db_dir,
            )
        else:
            self.vectorstore.add_documents(chunks)

        # 持久化
        self.vectorstore.persist()
        print(f"向量索引已保存到 Chroma: {self.chroma_db_dir}")

        return len(chunks)

    def load_vectorstore(self) -> bool:
        """
        从磁盘加载已有的 Chroma 向量存储

        Returns:
            是否成功加载
        """
        if not os.path.exists(self.chroma_db_dir):
            print("未找到 Chroma 数据库目录")
            return False

        try:
            embeddings = self._get_embeddings()
            self.vectorstore = Chroma(
                persist_directory=self.chroma_db_dir,
                embedding_function=embeddings,
            )
            print("Chroma 向量索引加载成功")
            return True
        except Exception as e:
            print(f"加载向量索引失败: {e}")
            return False

    def _init_memory(self):
        """初始化对话记忆"""
        if self.memory is None:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )

    def _build_qa_chain(self, k: int = DEFAULT_TOP_K):
        """
        构建 ConversationalRetrievalChain 问答链
        任务要求：使用 LangChain 的 ConversationalRetrievalChain 将检索器和 Ollama 大模型连接

        Args:
            k: 检索的文档数量（默认3）
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化，请先索引文档或加载已有索引")

        llm = self._get_llm()
        self._init_memory()

        # 自定义提示模板
        custom_template = SYSTEM_PROMPT

        # 创建 ConversationalRetrievalChain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(custom_template)},
            return_source_documents=True,
            verbose=False,
        )

    def query(self, question: str, k: int = DEFAULT_TOP_K) -> dict:
        """
        对文档进行问答（支持多轮对话）
        任务要求：返回最相关的3个文本块

        Args:
            question: 用户问题
            k: 检索的文档数量（默认3）

        Returns:
            包含答案、来源文档和对话历史的字典
        """
        if self.qa_chain is None:
            self._build_qa_chain(k=k)

        result = self.qa_chain.invoke({"question": question})
        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", []),
            "chat_history": result.get("chat_history", []),
        }

    def get_relevant_chunks(self, query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
        """
        检索与查询最相关的文本块
        任务要求：实现检索函数，返回最相关的3个文本块

        Args:
            query: 查询文本
            k: 返回的文本块数量（默认3）

        Returns:
            相关文档列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")

        return self.vectorstore.similarity_search(query, k=k)

    def clear_memory(self):
        """清除对话记忆"""
        if self.memory is not None:
            self.memory.clear()
            print("对话记忆已清除")

    def get_index_info(self) -> Optional[dict]:
        """
        获取当前索引信息

        Returns:
            索引信息字典，如果没有索引则返回 None
        """
        if self.vectorstore is None:
            # 尝试加载
            if not self.load_vectorstore():
                return None

        try:
            # 获取集合中的文档数量
            collection = self.vectorstore._collection
            doc_count = collection.count()
            return {
                "indexed": True,
                "doc_count": doc_count,
                "db_path": self.chroma_db_dir,
            }
        except Exception as e:
            print(f"获取索引信息失败: {e}")
            return None

    def clear_index(self):
        """清除所有索引数据"""
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
                self.vectorstore = None
                print("向量索引已清除")
            except Exception as e:
                print(f"清除索引失败: {e}")

        # 清除对话记忆
        self.clear_memory()


# ==================== 命令行版本 RAG 问答脚本 ====================

def run_cli_rag():
    """
    命令行版 RAG 问答脚本
    任务要求：将前两个任务的代码整合成一个完整的 RAG 问答脚本
    """
    print("=" * 50)
    print("RAG 智能问答系统 - 命令行版")
    print("=" * 50)

    # 初始化 RAG 管道
    pipeline = RAGPipeline()

    # 检查是否已有索引
    if not pipeline.load_vectorstore():
        print("\n未找到现有索引，开始构建知识库...")
        # 索引 docs 目录下的文档
        docs_dir = os.path.join(os.path.dirname(__file__), "docs")
        if os.path.exists(docs_dir):
            count = pipeline.index_documents_from_directory(docs_dir)
            print(f"知识库构建完成，共索引 {count} 个文本块")
        else:
            print(f"文档目录不存在: {docs_dir}")
            return

    # 显示索引信息
    info = pipeline.get_index_info()
    if info:
        print(f"\n当前知识库状态:")
        print(f"  - 文本块数量: {info['doc_count']}")
        print(f"  - 数据库路径: {info['db_path']}")

    print("\n" + "=" * 50)
    print("开始问答（输入 'quit' 退出，输入 'clear' 清除记忆）")
    print("=" * 50 + "\n")

    # 测试问答循环
    while True:
        question = input("\n🤔 请输入问题: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break

        if question.lower() == "clear":
            pipeline.clear_memory()
            continue

        if not question:
            continue

        try:
            print("\n🤖 正在思考...")
            result = pipeline.query(question)

            print(f"\n💡 回答:\n{result['answer']}")

            # 显示来源
            if result.get("source_documents"):
                print(f"\n📚 参考来源（共 {len(result['source_documents'])} 个）:")
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get("source", "未知来源")
                    print(f"  [{i}] {source}")

        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    run_cli_rag()
