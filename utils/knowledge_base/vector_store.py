"""
向量存储服务模块 - 提供文档向量化、存储和检索功能
"""
from typing import List, Optional
from pathlib import Path
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from utils.decorators import error_handler
from utils.logger.logger_manager import singleton_logger

# 导入配置项
from settings.system_settings import (
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    MAX_RETRIEVED_DOCS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_PATH,
    DEFAULT_SIMILARITY_THRESHOLD
)


class VectorStoreService:
    """
    向量存储服务类，用于管理文档向量存储

    核心功能：
    - 初始化向量存储环境
    - 创建/加载/保存向量索引
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
        self.index_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        self.vector_index = None                                  # LlamaIndex向量索引实例

        # 全局设置
        Settings.embed_model = self._create_embedding_model()
        Settings.node_parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # 尝试加载现有索引
        self.load_vector_store()

        singleton_logger.info(f"向量存储服务初始化完成 | 索引目录: {self.index_dir}")

    def _create_embedding_model(self) -> BaseEmbedding:
        """
        创建Ollama嵌入模型实例

        Returns:
            BaseEmbedding: Ollama嵌入模型实例
        """
        return OllamaEmbedding(
            model_name=EMBEDDING_MODEL,
            base_url=EMBEDDING_BASE_URL,
            ollama_additional_kwargs={"mirostat": 0},
            timeout=60
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
            if Settings.embed_model.model_name != model_name:
                # 更新全局嵌入模型
                Settings.embed_model = OllamaEmbedding(
                    model_name=model_name,
                    base_url=EMBEDDING_BASE_URL
                )

                # 重新加载索引以应用新模型
                self.load_vector_store()

                singleton_logger.info(f"嵌入模型已更新为: {model_name}")
                return True
            return False
        except Exception as e:
            singleton_logger.error(f"更新嵌入模型失败: {str(e)}")
            return False

    @error_handler()
    def create_vector_store(self, nodes: List[BaseNode]) -> Optional[VectorStoreIndex]:
        """
        创建全新的向量存储（覆盖现有索引）

        Args:
            nodes: 预处理后的文档块列表

        Returns:
            Optional[VectorStoreIndex]: 创建成功的向量索引实例，失败时返回None
        """
        if not nodes:
            singleton_logger.warning("没有文档块可以创建向量存储")
            return None

        singleton_logger.info(f"开始创建向量存储，文档块数量: {len(nodes)}")

        try:
            # 创建存储上下文
            storage_context = StorageContext.from_defaults()

            # 使用文档块创建新索引
            self.vector_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True
            )

            # 持久化保存索引
            self._save_vector_store()

            singleton_logger.info(f"向量存储创建成功，包含 {len(nodes)} 个文档块")
            return self.vector_index
        except Exception as e:
            singleton_logger.error(f"创建向量存储失败: {str(e)}")
            return None

    def _save_vector_store(self):
        """
        内部方法：保存向量存储到磁盘
        """
        if not self.vector_index:
            singleton_logger.warning("没有索引可保存")
            return

        try:
            # 确保存储目录存在
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # 保存索引
            self.vector_index.storage_context.persist(
                persist_dir=self.index_dir)
            singleton_logger.info(f"向量存储已保存到: {self.index_dir}")
        except Exception as e:
            singleton_logger.error(f"保存向量存储失败: {str(e)}")

    @error_handler()
    def load_vector_store(self) -> Optional[VectorStoreIndex]:
        """
        从磁盘加载向量存储

        Returns:
            Optional[VectorStoreIndex]: 加载成功的向量索引实例，失败时返回None
        """
        try:
            # 检查索引文件是否存在
            if (self.index_dir / "docstore.json").exists():
                # 重建存储上下文
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.index_dir)
                # 加载索引
                self.vector_index = load_index_from_storage(storage_context)
                singleton_logger.info("向量存储加载成功")
                return self.vector_index
            singleton_logger.warning("向量存储文件不存在")
        except Exception as e:
            singleton_logger.error(f"加载向量存储失败: {str(e)}")
        return None

    @error_handler()
    def search_documents(self, query: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> List[BaseNode]:
        """
        执行相似性搜索并返回文档块

        Args:
            query: 查询文本
            threshold: 相似度分数阈值（默认使用系统设置）

        Returns:
            List[BaseNode]: 符合阈值条件的相关文档块列表
        """
        # 确保索引已加载
        if not self.vector_index:
            self.vector_index = self.load_vector_store()
            if not self.vector_index:
                singleton_logger.warning("向量存储未初始化")
                return []

        try:
            # 创建检索器
            retriever = VectorIndexRetriever(
                index=self.vector_index,
                similarity_top_k=MAX_RETRIEVED_DOCS
            )

            # 执行检索
            retrieved_nodes = retriever.retrieve(query)

            # 应用相似度阈值过滤
            filtered_nodes = [
                node for node in retrieved_nodes
                if node.score and node.score > threshold
            ]

            singleton_logger.info(
                f"搜索到 {len(filtered_nodes)} 个相关文档块，相似度阈值: {threshold}")
            return filtered_nodes

        except Exception as e:
            singleton_logger.error(f"搜索文档块失败: {str(e)}")
            return []

    def get_context(self, nodes: List[BaseNode]) -> str:
        """
        将文档块列表合并为连续文本上下文

        Args:
            nodes: 文档块列表

        Returns:
            str: 合并后的上下文字符串
        """
        return "\n\n".join(node.get_content() for node in nodes) if nodes else ""

    def clear_index(self):
        """清除所有索引文件并重置向量存储"""
        try:
            # 删除索引目录所有文件
            for file in self.index_dir.glob("*"):
                file.unlink()
            # 重置索引实例
            self.vector_index = None
            singleton_logger.info("索引已清除")
        except Exception as e:
            singleton_logger.error(f"清除索引失败: {str(e)}")
            raise

    def add_documents(self, nodes: List[BaseNode]):
        """
        向现有索引添加新文档块

        Args:
            nodes: 要添加的文档块列表
        """
        if not self.vector_index:
            singleton_logger.warning("索引未初始化，将创建新索引")
            return self.create_vector_store(nodes)

        try:
            # 向索引添加新文档块
            for node in nodes:
                self.vector_index.insert_nodes([node])

            # 保存更新后的索引
            self._save_vector_store()

            singleton_logger.info(f"成功添加 {len(nodes)} 个新文档块到索引")
            return True
        except Exception as e:
            singleton_logger.error(f"添加文档失败: {str(e)}")
            return False

    def get_doc_count(self) -> int:
        """
        获取当前索引中的文档块数量

        Returns:
            int: 文档块数量，如果索引未加载则返回0
        """
        if self.vector_index is None:
            return 0
        try:
            # 通过storage_context获取文档存储
            docstore = self.vector_index.storage_context.docstore
            # 获取所有文档节点的字典，并计算长度
            return len(docstore.docs)
        except Exception as e:
            singleton_logger.error(f"获取文档数量失败: {e}")
            return 0
