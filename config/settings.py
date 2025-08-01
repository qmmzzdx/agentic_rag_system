"""
系统配置文件 - 包含所有常量和核心配置项
"""

# ---------------------- 文件路径配置 ----------------------
VECTOR_STORE_PATH = "faiss_index"       # 向量存储索引目录

# ---------------------- 模型服务配置 ----------------------
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

# ---------------------- RAG 配置 ----------------------
DEFAULT_SIMILARITY_THRESHOLD = 0.7  # 文档检索相似度阈值
MAX_RETRIEVED_DOCS = 3              # 最大检索文档数量

# ---------------------- LangChain 文本处理配置 ----------------------
CHUNK_SIZE = 300      # 文本分割块大小（字符数）
CHUNK_OVERLAP = 30    # 文本分割重叠区域大小（字符数）
SEPARATORS = [
    "\n\n", "\n", "。", "！", "？",
    ".", "!", "?", " ", ""
]  # 文本分割符优先级列表

# ---------------------- 高德地图API配置 ----------------------
# 高德开发者平台申请的Web服务API Key
# 申请地址：https://lbs.amap.com/api/webservice/guide/create-project/get-key
AMAP_API_KEY = "xxx"  # 高德地图API密钥，根据使用请自行申请

# 天气查询API接口地址
# 文档：https://lbs.amap.com/api/webservice/guide/api/weatherinfo
# 支持查询实时天气和未来天气预报
AMAP_WEATHER_API_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

# 地理编码API接口地址
# 文档：https://lbs.amap.com/api/webservice/guide/api/georegeo
# 用于将地址转换为经纬度坐标或获取行政区划编码(adcode)
AMAP_GEO_API_URL = "https://restapi.amap.com/v3/geocode/geo"

# ---------------------- 对话历史管理配置 ----------------------
MAX_HISTORY_TURNS = 5  # 最大对话轮次保留数
