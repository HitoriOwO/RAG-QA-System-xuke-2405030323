# RAG-QA-System-xuke-2405030323

## 项目简介

基于检索增强生成（RAG）技术的智能文档问答系统。用户可以上传 PDF、TXT、Markdown 等格式的文档，系统会自动对文档进行向量化索引，然后利用本地部署的大语言模型（通过 Ollama）对用户提出的问题进行精准回答，并展示参考来源。

## 环境要求与安装步骤

### 环境要求

- **Python**: 3.9 或以上版本
- **Ollama**: 最新版本（用于本地运行大语言模型）
- **操作系统**: Windows / macOS / Linux

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/HitoriOwO/RAG-QA-System-xuke-2405030323.git
cd RAG-QA-System-xuke-2405030323
```

#### 2. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt
```

#### 3. 安装 Ollama 并下载模型

**安装 Ollama：**

前往 [Ollama 官网](https://ollama.com/) 下载并安装对应操作系统的版本。

安装完成后，在终端中验证安装：

```bash
ollama --version
```

**下载所需模型：**

```bash
# 下载大语言模型（用于生成回答）
ollama pull qwen2.5:7b

# 下载嵌入模型（用于文本向量化）
ollama pull nomic-embed-text
```

> **提示：** 如果需要使用其他模型，可以在应用侧边栏中修改模型名称。推荐的大语言模型包括 `qwen2.5:7b`、`llama3`、`gemma2` 等；推荐的嵌入模型包括 `nomic-embed-text`、`mxbai-embed-large` 等。

## 使用说明

### 1. 启动 Ollama 服务

确保 Ollama 服务正在运行：

```bash
ollama serve
```

> 通常安装 Ollama 后，服务会自动在后台运行（默认端口 11434）。

### 2. 运行 Web 应用

```bash
streamlit run app.py
```

应用启动后，浏览器会自动打开 `http://localhost:8501`。

### 3. 上传文档

1. 在左侧边栏找到 **"文档管理"** 区域
2. 点击 **"上传文档"** 按钮，选择 PDF、TXT、Markdown 或 DOCX 格式的文件
3. 可调整 **文本块大小** 和 **重叠大小** 参数
4. 点击 **"索引文档"** 按钮，等待索引完成

### 4. 开始提问

1. 索引完成后，在底部输入框中输入您的问题
2. 系统会检索相关文档内容并生成回答
3. 点击 **"查看参考来源"** 可查看回答所依据的原文片段

## 项目结构

```
RAG-QA-System/
├── app.py                 # Streamlit Web 应用主入口
├── config.py              # 配置文件（模型、路径等参数）
├── rag_pipeline.py        # RAG 核心流程（索引构建、问答）
├── document_loader.py     # 文档加载与文本切分
├── embeddings.py          # 向量嵌入模块
├── requirements.txt       # Python 依赖列表
├── .gitignore             # Git 忽略规则
├── README.md              # 项目说明文档
├── docs/
│   └── sample_document.md # 示例文档
├── utils/
│   ├── __init__.py
│   └── helpers.py         # 工具函数
├── uploads/               # 上传文件目录（运行时生成）
├── vectorstore/           # 向量存储目录（运行时生成）
└── screenshots/           # 项目截图
```

## 关键技术点说明

### RAG 流程

本项目采用经典的 RAG（Retrieval-Augmented Generation）流程：

1. **文档加载**：使用 LangChain 的 Document Loaders 加载多种格式的文档（PDF、TXT、Markdown、DOCX）
2. **文本切分**：使用 `RecursiveCharacterTextSplitter` 将长文档切分为较小的文本块（默认 500 字符/块，50 字符重叠）
3. **向量化**：使用 Ollama 的 `nomic-embed-text` 模型将文本块转换为向量表示
4. **向量存储**：使用 FAISS（Facebook AI Similarity Search）构建高效的向量索引
5. **相似度检索**：用户提问时，将问题向量化，在 FAISS 索引中检索最相关的 K 个文本块
6. **增强生成**：将检索到的相关文本块作为上下文，连同用户问题一起输入到 LLM 中生成回答

### 所用模型

| 用途 | 模型 | 说明 |
|------|------|------|
| 大语言模型（LLM） | qwen2.5:7b | 阿里云通义千问，中文能力强 |
| 嵌入模型 | nomic-embed-text | 高质量文本嵌入模型 |

### 技术栈

- **Web 框架**: Streamlit
- **LLM 框架**: LangChain
- **向量数据库**: FAISS
- **本地模型服务**: Ollama
- **嵌入方式**: Ollama Embeddings API

## 项目效果截图

### 1. 主界面 - 文档上传与索引

![主界面](screenshots/screenshot_1.png)

*上传文档并构建向量索引的界面*

### 2. 问答界面 - 智能问答

![问答界面](screenshots/screenshot_2.png)

*基于文档内容的智能问答交互*

### 3. 参考来源展示

![参考来源](screenshots/screenshot_3.png)

*展示回答所依据的文档原文片段*

> **注意：** 请在运行项目后截取实际界面截图，替换 `screenshots/` 目录下的占位图片。

## 已知问题与改进方向

### 已知问题

- 首次加载嵌入模型时可能较慢，取决于模型大小和硬件性能
- 对于非常大的文档（如超过 100 页的 PDF），索引构建时间较长
- 当前仅支持单轮独立问答，暂不支持多轮对话上下文记忆

### 改进方向

- [ ] 支持多轮对话，保持上下文记忆
- [ ] 添加更多文档格式支持（如 PPT、Excel）
- [ ] 引入重排序（Reranking）机制提升检索精度
- [ ] 添加对话历史记录导出功能
- [ ] 支持同时索引多个文档并进行跨文档问答
- [ ] 添加流式输出，提升回答生成体验

## 许可证

MIT License
