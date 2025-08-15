# Agentic RAG System 智能问答系统(Version 0.1) 🤖

基于Agno多智能体框架和Llama_index的本地RAG(检索增强生成)智能问答系统，支持文档问答和天气查询功能，提供高效便捷的自然语言交互体验。

## 系统亮点 ✨

- **📚 强大的问答能力**：基于Qwen3和DeepSeek-R1大模型，提供高质量的对话能力
- **🔍 本地RAG检索增强**：上传文档后可针对文档内容进行智能问答
- **☀️ 实时天气查询**：集成高德地图API，支持查询全国城市天气
- **💬 多模式对话**：支持RAG文档问答模式和普通对话模式
- **⚙️ 灵活配置**：可选择不同的模型版本和嵌入模型
- **🖥️ 用户友好界面**：基于Streamlit构建的简洁易用界面

## 技术架构 🏗️

- **🧠 大模型引擎** ：基于Qwen3和DeepSeek-R1系列模型，支持本地部署，提供强大的自然语言理解与生成能力  
- **⚡ 框架基础**：采用agno框架构建，具备灵活的代理能力，支持多任务协同处理  
- **🗃️ 向量数据库**：FAISS高效向量检索系统，实现毫秒级文档相似度匹配  
- **🔢 嵌入模型**：默认集成Qwen3-Embedding嵌入模型，支持多种文本向量化方案自由切换  
- **🌐 Web框架**：Streamlit轻量级Web界面，零前端代码实现交互式应用  
- **🧰 工具能力**：可扩展的工具集成体系，已内置高德天气等实用功能  
- **📄 文档处理**：智能文档解析引擎，支持PDF/TXT等多种格式自动分块处理  

## 系统快速部署

### 环境要求

- Python 3.12
- Ollama (用于本地部署大模型)
- NVIDIA GPU 24GB (推荐，非必需)
- FAISS

### 安装步骤

#### 1. 安装 Ollama

参考 [Ollama 官网](https://ollama.com/) 以获取安装和配置 Ollama 的详细步骤。

```bash
# 安装Qwen3模型
ollama pull qwen3:8b

# 安装嵌入模型
ollama run dengcao/Qwen3-Embedding-8B:Q5_K_M
```

#### 2. 添加要用的API_KEY

在`settings/system_settings.py`中添加要用的API_KEY：

```python
AMAP_API_KEY = "xxxx"
```

#### 3. 启动应用

```bash
streamlit run ./app.py --server.port 6006
```

## Agentic RAG 项目结构

```
agentic_rag_system/
├── agent/                             # 智能体相关模块
│   ├── agent_prompt/                  # 智能体提示词配置
│   │   └── agent_instructions.py      # 智能体指令和提示模板
│   ├── tools/                         # 智能体工具集
│   │   ├── agent_tools_select.py      # 工具选择与路由逻辑
│   │   └── weather_tool.py            # 天气查询工具实现
│   └── chat_agent.py                  # 智能体核心实现
├── settings/                          # 系统配置目录
│   └── system_settings.py             # 系统级配置和常量定义
├── utils/                             # 工具与辅助模块
│   ├── chat_record/                   # 聊天记录管理
│   │   └── chat_history.py            # 对话历史存储与处理
│   ├── document_processor/            # 文档处理模块
│   │   └── doc_processor.py           # 文档加载与预处理
│   ├── knowledge_base/                # 知识库管理
│   │   └── vector_store.py            # 向量存储服务实现
│   ├── logger/                        # 日志管理模块
│   │   └── logger_manager.py          # 日志记录器实现
│   ├── ui/                            # 用户界面组件
│   │   └── ui_components.py           # Streamlit UI组件库
│   └── decorators.py                  # Python装饰器工具
├── app.py                             # Streamlit应用主入口
├── LICENSE
└── README.md
```

## 核心功能

### 1. 文档问答 (RAG模式)

- **上传文档**：支持PDF、TXT等多种格式文档  
- **自动处理**：系统自动处理文档并构建向量索引  
- **智能提问**：询问与文档相关的问题  
- **精准回答**：系统检索相关内容并生成准确回答  

### 2. 普通对话模式

- **自由切换**：可在侧边栏直接切换对话模式  
- **自然对话**：直接与模型进行自然语言交流  
- **通用能力**：利用大模型的通用知识库能力  

### 3. 天气查询功能

在任意模式下均可查询全国各地的实时天气情况：

```text
北京今天天气怎么样？
上海明天会下雨吗？
广州最近一周的天气如何？
```

## 系统配置

在侧边栏可灵活配置系统参数：

- **模型选择**：可选不同大小的Qwen3和DeepSeek模型（qwen3:0.6b，deepseek-r1:latest，qwen3:latest等）  
- **嵌入模型**：可选不同的文本嵌入模型（Qwen3-Embedding, BGE-M3等）  
- **相似度阈值**：调整文档检索的相似度要求（默认0.7）  
- **RAG模式开关**：快速切换RAG文档问答模式和普通对话模式  

## 使用提示

- 上传文档后，系统会自动处理构建索引，大型文档请耐心等待  
- 更改嵌入模型后，可能需要重新处理文档以更新索引  
- 对于查询效果不佳的情况，尝试调整相似度阈值  
- 天气查询功能需要网络连接以访问高德地图API  
- 聊天历史自动保存在`./chat_history/ars-chat-history-YYYYMMDD.json`中，重启后仍可访问  

## 开发者指南

### 核心组件

- **App类**：主应用类，管理整体流程和UI渲染  
- **RAGAgent**：封装大模型交互和工具调用  
- **VectorStoreService**：管理文档向量存储和检索  
- **DocumentProcessor**：处理和分块各种格式的文档  
- **WeatherTools**：提供天气查询功能的工具类  
- **ChatHistoryManager**：管理对话历史的持久化存储  
- **UIComponents**：提供可复用的UI渲染组件 

## 扩展方法

### 1. 添加新工具

参考`agent_tools_select.py`实现新工具，在`agent_tools_select.py`中添加类：

```python
# 工具列表，包含所有可用的工具实例
# 用于在agent中统一管理和调用各种工具
TOOL_LISTS = [
    MathTool(),      # 数学计算工具实例
    WeatherTool()    # 天气查询工具实例
]
```

### 2. 支持新文档格式

在`./utils/document_processor/__init__.py`中添加新的文档加载器：

```python
# 自动映射扩展名到对应的读取器
EXTENSION_READER_MAP: Dict[str, Type] = {
    # 文本类
    '.txt': FlatReader,
    # csv类
    '.csv': CSVReader,
    # markdown类
    '.md': MarkdownReader,
    # docx类
    '.docx': DocxReader,
    # 文档类
    '.pdf': OCRPDFReader
}
```

### 3. 自定义配置

在`settings/system_settings.py`中添加新的模型和其余配置：
```python
# 默认模型
DEFAULT_MODEL = "qwen3:0.6b"
# 可用模型列表
AVAILABLE_MODELS = ["qwen3:0.6b", "deepseek-r1:latest", "qwen3:latest"]

# 嵌入模型配置
EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
# 嵌入模型服务地址
EMBEDDING_BASE_URL = "http://localhost:11434"
# 可用嵌入模型列表
# 这些模型用于将文本转换为向量表示
# 可根据需要添加或修改
AVAILABLE_EMBEDDING_MODELS = [
    "dengcao/Qwen3-Embedding-0.6B:Q8_0",
    "dengcao/Qwen3-Embedding-8B:Q5_K_M",
    "bge-m3:latest",
]
```

