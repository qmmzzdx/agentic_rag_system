# Agentic RAG System 智能问答系统 🤖

基于Qwen3和DeepSeek-R1的本地RAG(检索增强生成)智能问答系统，支持文档问答和天气查询功能，提供高效便捷的自然语言交互体验。

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

在`config/settings.py`中添加要用的API_KEY：

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
├── config/                       # 配置文件目录
│   ├── agent_instructions.py     # 智能体指令和提示模板配置
│   ├── agent_tools.py            # 工具配置 - 定义智能体可使用的各种工具
│   └── settings.py               # 系统配置文件 - 包含所有常量和核心配置项
├── models/                       # 模型相关代码
│   └── agent.py                  # RAG智能体实现
├── services/                     # 核心服务工具
│   └── weather_tools.py          # 天气查询工具
├── utils/                        # 工具文件目录
│   ├── chat_history.py           # 对话历史管理类 - 提供对话历史的存储、加载、格式化和导出功能
│   ├── decorators.py             # 装饰器工具模块
│   ├── document_processor.py     # 文档处理模块 - 提供多种文档类型的加载、处理和缓存功能
│   ├── logger_manager.py         # 日志管理类 - 提供全局单例日志记录器，支持控制台和文件输出
│   ├── ui_components.py          # UI组件模块，包含所有Streamlit UI渲染逻辑
│   └── vector_store.py           # 向量存储服务模块 - 提供文档向量化、存储和检索功能
├── app.py                        # 主应用入口
└── README.md                     # 项目文档
```

## 核心功能

### 1. 文档问答 (RAG模式)

- **上传文档**：支持PDF、TXT、DOCX等多种格式文档  
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

参考`weather_tools.py`实现新工具，在`agent_tools.py`中添加：

```python
# 天气查询工具
TOOL_WEATHER = {
    "name": "query_weather",
    "description": "查询指定城市的天气预报",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "要查询的城市名称"
            }
        },
        "required": ["city"]
    },
    "entrypoint": WeatherTools(AMAP_API_KEY).query_weather
}

# 可以继续添加其他工具...
# TOOL_CALCULATOR = {...}
```

### 2. 支持新文档格式

在`document_processor.py`中添加新的文档加载器：

```python
# 支持的文件类型和对应加载器
SUPPORTED_EXTENSIONS: Dict[str, Type[BaseLoader]] = {
    '.pdf': PyPDFLoader,   # PDF文档加载器
    '.txt': TextLoader,    # TXT文件加载器
}
```

### 3. 自定义配置

在`config/settings.py`中添加新的模型和其余配置：
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
