# RAG-QA-System-xuke-2405030323

## 项目简介

基于检索增强生成（RAG）技术的智能文档问答系统。用户可以上传 PDF、DOCX 等格式的文档，系统会自动对文档进行向量化索引并存储到 Chroma 向量数据库，然后利用本地部署的大语言模型（通过 Ollama）对用户提出的问题进行精准回答，支持多轮对话和会话记忆功能。

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
ollama pull deepseek-r1:7b
# 或 ollama pull qwen2:7b

# 下载嵌入模型（用于文本向量化）
ollama pull nomic-embed-text
# 或 ollama pull all-minilm
```

**验证模型安装：**

```bash
python test_ollama.py
```

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

### 3. 上传文档并构建知识库

1. 在左侧边栏找到 **"文档管理"** 区域
2. 点击 **"上传文档"** 按钮，选择 PDF 或 DOCX 格式的文件（支持多文件上传）
3. 可调整 **文本块大小**（默认 1000）和 **重叠大小**（默认 200）参数
4. 点击 **"构建知识库"** 按钮，等待索引完成

### 4. 开始提问

1. 知识库构建完成后，在底部输入框中输入您的问题
2. 系统会检索相关文档内容并生成回答
3. 点击 **"查看参考来源"** 可查看回答所依据的原文片段
4. 支持多轮对话，系统会记住对话上下文

### 5. 命令行版本

也可以使用命令行版本进行问答：

```bash
python rag_pipeline.py
```

## 项目结构

```
RAG-QA-System/
├── app.py                 # Streamlit Web 应用主入口
├── config.py              # 配置文件（模型、路径等参数）
├── rag_pipeline.py        # RAG 核心流程（索引构建、问答）
├── document_loader.py     # 文档加载与文本切分
├── embeddings.py          # 向量嵌入模块
├── test_ollama.py         # Ollama API 测试脚本
├── build_exe.py           # PyInstaller 打包脚本
├── requirements.txt       # Python 依赖列表
├── .gitignore             # Git 忽略规则
├── README.md              # 项目说明文档
├── docs/                  # 预置文档目录（5份NLP相关文档）
│   ├── 01_自然语言处理概述.md
│   ├── 02_Transformer架构详解.md
│   ├── 03_预训练语言模型.md
│   ├── 04_词嵌入技术.md
│   └── 05_RAG检索增强生成.md
├── utils/                 # 工具函数
│   ├── __init__.py
│   └── helpers.py
├── uploads/               # 上传文件目录（运行时生成）
└── chroma_db/             # Chroma 向量数据库目录（运行时生成）
```

## 关键技术点说明

### RAG 流程

本项目采用完整的 RAG（Retrieval-Augmented Generation）流程：

1. **文档加载**：使用 LangChain 的 Document Loaders 加载多种格式的文档（PDF、DOCX、TXT、Markdown）
2. **批量处理**：支持批量读取指定文件夹内所有文档
3. **文本切分**：使用 `RecursiveCharacterTextSplitter`，`chunk_size=1000`，`chunk_overlap=200`
4. **向量化**：使用 Ollama 的 `nomic-embed-text` 或 `all-minilm` 模型将文本块转换为向量
5. **向量存储**：使用 **Chroma** 向量数据库存储和检索向量
6. **相似度检索**：用户提问时，检索最相关的 **3** 个文本块
7. **增强生成**：使用 `ConversationalRetrievalChain` 连接检索器和 LLM，支持多轮对话

### 所用模型

| 用途 | 模型 | 说明 |
|------|------|------|
| 大语言模型（LLM） | deepseek-r1:7b / qwen2:7b | 本地部署的大语言模型 |
| 嵌入模型 | nomic-embed-text / all-minilm | 文本向量化模型 |

### 技术栈

- **Web 框架**: Streamlit
- **LLM 框架**: LangChain
- **对话链**: ConversationalRetrievalChain（支持会话记忆）
- **向量数据库**: Chroma
- **本地模型服务**: Ollama
- **嵌入方式**: Ollama Embeddings API

### 系统提示词设计

```
你是一个专业的问答助手。请严格根据以下提供的参考文档内容来回答用户的问题。

重要规则：
1. 只能基于提供的参考文档内容回答
2. 如果文档中没有相关信息，必须明确说"文档中未找到相关答案"
3. 不要编造答案，不要添加文档中没有的信息
4. 回答要清晰、准确、简洁
```

## 项目效果截图

### 1. 主界面 - 知识库构建

![主界面](screenshots/screenshot_1.png)

*上传文档并构建知识库的界面*

### 2. 问答界面 - 智能问答

![问答界面](screenshots/screenshot_2.png)

*基于文档内容的智能问答交互，显示回答和参考来源*

### 3. 多轮对话展示

![多轮对话](screenshots/screenshot_3.png)

*支持多轮对话，系统记住上下文进行连续问答*

> **注意：** 请在运行项目后截取实际界面截图，替换 `screenshots/` 目录下的占位图片。

## 测试用例示例

### 与文档内容相关的问题（应能正确回答）

1. **什么是自然语言处理？**
   - 预期：解释 NLP 的定义和主要任务

2. **Transformer 的核心组件有哪些？**
   - 预期：说明自注意力机制、多头注意力、位置编码等

3. **BERT 和 GPT 有什么区别？**
   - 预期：解释两者在架构和预训练任务上的差异

4. **Word2Vec 的两种架构是什么？**
   - 预期：说明 CBOW 和 Skip-gram

5. **RAG 的优势有哪些？**
   - 预期：知识实时更新、减少幻觉、可溯源等

### 与文档内容无关的问题（应回答"未找到相关答案"）

1. **如何制作红烧肉？**
   - 预期：文档中未找到相关答案

2. **明天的天气怎么样？**
   - 预期：文档中未找到相关答案

## 打包为可执行文件

### 使用启动脚本（推荐）

```bash
python build_exe.py
# 选择选项 1，生成 启动RAG系统.bat
```

将项目文件和 `启动RAG系统.bat` 一起复制到目标电脑，双击运行即可。

### 使用 PyInstaller 打包（实验性）

```bash
python build_exe.py
# 选择选项 2，生成独立的 exe 文件
```

> **注意**：由于 Streamlit 的特殊架构，完全打包为独立 exe 较为复杂，生成的文件可能较大。

## 已知问题与改进方向

### 已知问题

- 首次加载嵌入模型时可能较慢，取决于模型大小和硬件性能
- 对于非常大的文档（如超过 100 页的 PDF），索引构建时间较长
- 当前仅支持单用户本地使用，暂不支持多用户并发

### 改进方向

- [ ] 支持更多文档格式（如 PPT、Excel、图片 OCR）
- [ ] 引入重排序（Reranking）机制提升检索精度
- [ ] 添加对话历史记录导出功能
- [ ] 支持同时索引多个文档并进行跨文档问答
- [ ] 添加流式输出，提升回答生成体验
- [ ] 实现多用户支持和权限管理
- [ ] 添加知识库备份和恢复功能

## 许可证

MIT License
