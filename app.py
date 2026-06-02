"""
Streamlit Web 应用主入口
提供文档上传、知识库构建和智能问答的交互界面
实现会话记忆（st.session_state），支持多轮对话
"""

import os
import sys
import time
import streamlit as st
from streamlit_chat import message

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    UPLOAD_DIR,
    CHROMA_DB_DIR,
    DOCS_DIR,
)
from rag_pipeline import RAGPipeline
from document_loader import get_document_info


def init_session_state():
    """初始化 Streamlit 会话状态"""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(
            llm_model=DEFAULT_LLM_MODEL,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            ollama_base_url=OLLAMA_BASE_URL,
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "index_ready" not in st.session_state:
        # 尝试加载已有索引
        st.session_state.index_ready = st.session_state.pipeline.load_vectorstore()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def render_sidebar():
    """渲染侧边栏 - 模型配置和文档管理"""
    with st.sidebar:
        st.title("⚙️ 系统配置")

        st.subheader("模型设置")
        llm_model = st.text_input(
            "LLM 模型名称",
            value=DEFAULT_LLM_MODEL,
            help="Ollama 中已安装的大语言模型名称（如：deepseek-r1:7b, qwen2:7b）",
        )
        embedding_model = st.text_input(
            "嵌入模型名称",
            value=DEFAULT_EMBEDDING_MODEL,
            help="Ollama 中已安装的嵌入模型名称（如：nomic-embed-text, all-minilm）",
        )
        ollama_url = st.text_input(
            "Ollama 服务地址",
            value=OLLAMA_BASE_URL,
            help="Ollama API 服务地址",
        )

        # 更新 pipeline 配置
        if st.button("🔄 应用配置", use_container_width=True):
            st.session_state.pipeline = RAGPipeline(
                llm_model=llm_model,
                embedding_model=embedding_model,
                ollama_base_url=ollama_url,
            )
            st.session_state.index_ready = st.session_state.pipeline.load_vectorstore()
            st.success("配置已更新！")
            st.rerun()

        st.divider()
        st.subheader("📄 文档管理")

        # 文档上传
        uploaded_files = st.file_uploader(
            "上传文档（支持 PDF、DOCX）",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="支持 PDF、DOCX 格式，可一次上传多个文件",
        )

        if uploaded_files:
            st.info(f"已选择 {len(uploaded_files)} 个文件")

            # 切分参数
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("文本块大小", 500, 2000, 1000, 100)
            with col2:
                chunk_overlap = st.slider("重叠大小", 0, 500, 200, 10)

            if st.button("📑 构建知识库", use_container_width=True, type="primary"):
                # 保存上传的文件
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                total_chunks = 0

                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner(f"正在索引: {uploaded_file.name}..."):
                        try:
                            count = st.session_state.pipeline.index_single_document(
                                save_path,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                            )
                            total_chunks += count
                            if uploaded_file.name not in st.session_state.indexed_files:
                                st.session_state.indexed_files.append(uploaded_file.name)
                        except Exception as e:
                            st.error(f"索引失败 {uploaded_file.name}: {e}")

                    progress_bar.progress((i + 1) / len(uploaded_files))

                st.session_state.index_ready = True
                st.success(f"知识库构建完成！共索引 {total_chunks} 个文本块")
                st.rerun()

        st.divider()

        # 知识库状态显示
        st.subheader("📊 知识库状态")
        info = st.session_state.pipeline.get_index_info()
        if info:
            st.success(f"✅ 知识库已就绪")
            st.write(f"- 文本块数量: **{info['doc_count']}**")
            if st.session_state.indexed_files:
                st.write(f"- 已索引文件:")
                for f in st.session_state.indexed_files[-5:]:  # 只显示最近5个
                    st.write(f"  - {f}")
        else:
            st.warning("⚠️ 尚未构建知识库")

        # 显示 docs 目录信息
        docs_info = get_document_info(DOCS_DIR)
        if docs_info and docs_info.get("file_counts", {}).get("total", 0) > 0:
            st.write(f"- 预置文档: **{docs_info['file_counts']['total']}** 个")

        if st.button("🗑️ 清除知识库", use_container_width=True):
            st.session_state.pipeline.clear_index()
            st.session_state.index_ready = False
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("知识库已清除")
            st.rerun()


def render_chat():
    """渲染聊天界面"""
    st.title("💬 RAG 智能问答系统")
    st.caption("基于检索增强生成（RAG）的文档问答系统，支持多轮对话")

    # 显示历史消息
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{i}")
        else:
            message(msg["content"], is_user=False, key=f"bot_{i}")
            # 显示来源文档
            if "sources" in msg and msg["sources"]:
                with st.expander("📎 查看参考来源"):
                    for j, doc in enumerate(msg["sources"]):
                        source = doc.metadata.get("source", "未知来源")
                        st.markdown(f"**来源 {j + 1}: {source}**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.divider()

    # 输入框
    if prompt := st.chat_input("请输入您的问题..."):
        if not st.session_state.index_ready:
            st.error("请先上传文档并构建知识库！")
            return

        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        message(prompt, is_user=True, key=f"user_{len(st.session_state.messages) - 1}")

        # 生成回答
        with st.spinner("🤔 正在思考..."):
            try:
                start_time = time.time()
                result = st.session_state.pipeline.query(prompt)
                elapsed = time.time() - start_time

                answer = result["answer"]
                sources = result.get("source_documents", [])

                # 添加助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

                message(answer, is_user=False, key=f"bot_{len(st.session_state.messages) - 1}")

                # 显示来源
                if sources:
                    with st.expander("📎 查看参考来源"):
                        for j, doc in enumerate(sources):
                            source = doc.metadata.get("source", "未知来源")
                            st.markdown(f"**来源 {j + 1}: {source}**")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.divider()

                st.caption(f"⏱️ 回答耗时: {elapsed:.2f} 秒")

            except Exception as e:
                error_msg = f"回答生成失败: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                message(error_msg, is_user=False, key=f"error_{len(st.session_state.messages) - 1}")


def main():
    """主函数"""
    st.set_page_config(
        page_title="RAG 智能问答系统",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
