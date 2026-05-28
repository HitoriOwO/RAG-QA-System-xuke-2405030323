"""
Streamlit Web 应用主入口
提供文档上传、索引构建和智能问答的交互界面
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
    VECTORSTORE_DIR,
)
from rag_pipeline import RAGPipeline


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


def render_sidebar():
    """渲染侧边栏 - 模型配置和文档管理"""
    with st.sidebar:
        st.title("⚙️ 系统配置")

        st.subheader("模型设置")
        llm_model = st.text_input(
            "LLM 模型名称",
            value=DEFAULT_LLM_MODEL,
            help="Ollama 中已安装的大语言模型名称",
        )
        embedding_model = st.text_input(
            "嵌入模型名称",
            value=DEFAULT_EMBEDDING_MODEL,
            help="Ollama 中已安装的嵌入模型名称",
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

        st.divider()
        st.subheader("📄 文档管理")

        # 文档上传
        uploaded_file = st.file_uploader(
            "上传文档",
            type=["pdf", "txt", "md", "docx"],
            help="支持 PDF、TXT、Markdown、DOCX 格式",
        )

        if uploaded_file is not None:
            # 保存上传的文件
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info(f"文件已保存: {uploaded_file.name}")

            # 切分参数
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("文本块大小", 200, 2000, 500, 100)
            with col2:
                chunk_overlap = st.slider("重叠大小", 0, 500, 50, 10)

            if st.button("📑 索引文档", use_container_width=True, type="primary"):
                with st.spinner(f"正在索引文档: {uploaded_file.name}..."):
                    try:
                        count = st.session_state.pipeline.index_documents(
                            save_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        st.session_state.indexed_files.append(uploaded_file.name)
                        st.session_state.index_ready = True
                        st.success(f"索引完成！共处理 {count} 个文本块")
                    except Exception as e:
                        st.error(f"索引失败: {e}")

        st.divider()

        # 索引信息
        st.subheader("📊 索引信息")
        info = st.session_state.pipeline.get_index_info()
        if info:
            st.success(f"✅ 索引已就绪")
            st.write(f"- 文本块数量: **{info['doc_count']}**")
            st.write(f"- 已索引文件:")
            for f in st.session_state.indexed_files:
                st.write(f"  - {f}")
        else:
            st.warning("⚠️ 尚未索引任何文档")

        if st.button("🗑️ 清除索引", use_container_width=True):
            if os.path.exists(VECTORSTORE_DIR):
                import shutil
                shutil.rmtree(VECTORSTORE_DIR)
            st.session_state.pipeline = RAGPipeline(
                llm_model=llm_model,
                embedding_model=embedding_model,
                ollama_base_url=ollama_url,
            )
            st.session_state.index_ready = False
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.success("索引已清除")


def render_chat():
    """渲染聊天界面"""
    st.title("💬 RAG 智能问答系统")
    st.caption("基于检索增强生成（RAG）的文档问答系统，使用 Ollama 本地大模型")

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
                        st.markdown(f"**来源 {j + 1}:**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.divider()

    # 输入框
    if prompt := st.chat_input("请输入您的问题..."):
        if not st.session_state.index_ready:
            st.error("请先上传并索引文档！")
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
                            st.markdown(f"**来源 {j + 1}:**")
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
