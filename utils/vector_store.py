"""
向量存储服务模块 - 提供文档向量化、存储和检索功能
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.decorators import error_handler

# 导入配置项
from config.settings import (
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    MAX_RETRIEVED_DOCS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    VECTOR_STORE_PATH
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    向量存储服务类，用于管理文档向量存储

    核心功能：
    - 初始化向量存储环境
    - 文档分块处理
    - 创建/加载/保存FAISS索引
    - 执行相似性搜索
    - 管理文档（添加/清除）
    """

    def __init__(self, index_dir: str = VECTOR_STORE_PATH):
        """
        初始化向量存储服务

        Args:
            index_dir: 索引文件存储目录路径（默认使用配置中的VECTOR_STORE_PATH）
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)  # 确保目录存在
        self.vector_store = None  # FAISS向量存储实例

        # 初始化Ollama文本嵌入模型
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=EMBEDDING_BASE_URL
        )

        # 初始化递归字符文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,        # 每个文本块的最大长度
            chunk_overlap=CHUNK_OVERLAP,  # 块之间的重叠字符数
            separators=SEPARATORS         # 用于分割文本的分隔符列表
        )

    def update_embedding_model(self, model_name: str) -> bool:
        """
        更新嵌入模型（热切换）

        Args:
            model_name: 新的嵌入模型名称

        Returns:
            bool: 是否成功更新模型
        """
        try:
            # 仅当模型名称不同时才更新
            if self.embeddings.model != model_name:
                self.embeddings = OllamaEmbeddings(
                    model=model_name,
                    base_url=EMBEDDING_BASE_URL
                )
                logger.info(f"嵌入模型已更新为: {model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"更新嵌入模型失败: {str(e)}")
            return False

    @error_handler()
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割为适合处理的文本块

        Args:
            documents: 原始文档列表

        Returns:
            List[Document]: 分割后的文档块列表
        """
        # 使用文本分割器进行分块处理
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(
            f"文档分块完成：原始文档数量 {len(documents)}，分块后文档数量 {len(split_docs)}")
        return split_docs

    @error_handler()
    def create_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        """
        创建全新的向量存储（覆盖现有索引）

        Args:
            documents: 待处理的文档列表

        Returns:
            Optional[FAISS]: 创建成功的FAISS实例，失败时返回None
        """
        if not documents:
            logger.warning("没有文档可以创建向量存储")
            return None

        logger.info(f"开始创建向量存储，原始文档数量: {len(documents)}")

        # 对文档进行分块处理
        split_documents = self.split_documents(documents)

        # 使用分块后的文档创建FAISS向量存储
        self.vector_store = FAISS.from_documents(
            split_documents,
            self.embeddings
        )

        # 保存新创建的向量存储
        self._save_vector_store(self.vector_store)

        logger.info(f"向量存储创建成功，包含 {len(split_documents)} 个文档块")
        return self.vector_store

    def _save_vector_store(self, vector_store: FAISS):
        """
        内部方法：保存向量存储到磁盘

        Args:
            vector_store: 要保存的FAISS实例
        """
        try:
            vector_store.save_local(str(self.index_dir))
            logger.info(f"向量存储已保存到: {self.index_dir}")
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")

    @error_handler()
    def load_vector_store(self) -> Optional[FAISS]:
        """
        从磁盘加载向量存储

        Returns:
            Optional[FAISS]: 加载成功的FAISS实例，失败时返回None
        """
        try:
            # 检查索引文件是否存在
            if (self.index_dir / "index.faiss").exists():
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True  # 允许加载可能不安全的序列化数据
                )
                logger.info("向量存储加载成功")
                return self.vector_store
            logger.warning("向量存储文件不存在")
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
        return None

    @error_handler()
    def search_documents(self, query: str, threshold: float = 0.7) -> List[Document]:
        """
        执行相似性搜索

        Args:
            query: 查询文本
            threshold: 相似度分数阈值（默认0.7）

        Returns:
            List[Document]: 符合阈值条件的相关文档列表
        """
        # 确保向量存储已加载
        if not self.vector_store:
            self.vector_store = self.load_vector_store()
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []

        try:
            # 执行带分数的相似性搜索
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=MAX_RETRIEVED_DOCS  # 最大返回结果数
            )

            # 根据阈值过滤结果（分数越高表示越不相似）
            results = [doc for doc,
                       score in docs_and_scores if score > threshold]

            logger.info(f"搜索到 {len(results)} 个相关文档，相似度阈值: {threshold}")
            return results

        except Exception as e:
            logger.error(f"搜索文档失败: {str(e)}")
            return []

    def get_context(self, docs: List[Document]) -> str:
        """
        将文档列表合并为连续文本上下文

        Args:
            docs: 文档列表

        Returns:
            str: 合并后的上下文字符串
        """
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)

    @error_handler()
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        添加单个文档到现有向量存储（增量更新）

        Args:
            content: 文档文本内容
            metadata: 文档元数据（可选）

        Returns:
            bool: 是否成功添加
        """
        if not content:
            logger.warning("文档内容为空，无法添加")
            return False

        try:
            # 创建Document对象
            doc = Document(page_content=content, metadata=metadata or {})

            # 对文档进行分块处理
            split_docs = self.split_documents([doc])

            # 确保向量存储已加载
            if not self.vector_store:
                self.vector_store = self.load_vector_store()
                if not self.vector_store:
                    # 不存在则创建新存储
                    self.vector_store = self.create_vector_store([doc])
                    return True

            # 向现有存储添加文档块
            self.vector_store.add_documents(split_docs)

            # 保存更新后的向量存储
            self._save_vector_store(self.vector_store)

            logger.info(
                f"成功添加文档，标题: {metadata.get('source', '未知') if metadata else '未知'}，分块数量: {len(split_docs)}")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False

    def clear_index(self):
        """清除所有索引文件并重置向量存储"""
        try:
            # 删除索引目录所有文件
            for file in self.index_dir.glob("*"):
                file.unlink()
            self.vector_store = None
            logger.info("索引已清除")
        except Exception as e:
            logger.error(f"清除索引失败: {str(e)}")
            raise
