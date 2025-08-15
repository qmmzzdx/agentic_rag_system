"""
文档处理模块 - 提供多种文档类型的加载、处理和缓存功能
"""
import hashlib
import json
import tempfile
from pathlib import Path
from typing import List, Optional

from utils.decorators import error_handler, log_execution
from utils.logger.logger_manager import singleton_logger

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

# 配置参数
from . import EXTENSION_READER_MAP
from settings.system_settings import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """
    文档处理器 - 实现文档加载、分块处理和智能缓存（基于LlamaIndex）

    关键特性：
    - 智能缓存：基于文件内容+处理参数生成唯一缓存键
    - 参数敏感：当分块参数变化时自动刷新缓存
    - 类型扩展：通过EXTENSION_READER_MAP支持新格式
    - 安全处理：使用临时文件确保资源清理
    """

    def __init__(self, cache_dir: str = "document_cache"):
        """
        初始化文档处理器

        Args:
            cache_dir: 缓存文件存储目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化文本分块器
        self.text_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        # 生成处理参数指纹（用于缓存键）
        self.params_fingerprint = self._generate_params_fingerprint()
        singleton_logger.info(f"文档处理器初始化完成 | 缓存目录: {self.cache_dir}")

    def _generate_params_fingerprint(self) -> str:
        """
        生成处理参数指纹 - 确保参数变化时刷新缓存

        指纹包含：
        - 分块大小 (CHUNK_SIZE)
        - 分块重叠量 (CHUNK_OVERLAP)

        Returns:
            str: 参数配置的MD5哈希值
        """
        params_str = f"""
            chunk_size={CHUNK_SIZE}
            chunk_overlap={CHUNK_OVERLAP}
        """.encode('utf-8')
        return hashlib.md5(params_str).hexdigest()

    def _generate_cache_key(self, file_content: bytes) -> str:
        """
        生成唯一缓存键 - 基于文件内容 + 处理参数

        核心逻辑：
        - 相同内容 + 相同参数 => 相同缓存键
        - 内容变化 | 参数变化 => 缓存键变化

        Args:
            file_content: 文件二进制内容

        Returns:
            str: 128位MD5哈希值
        """
        # 组合文件内容和参数指纹
        combined = file_content + self.params_fingerprint.encode('utf-8')
        return hashlib.md5(combined).hexdigest()

    def _get_cache_path(self, file_content: bytes) -> Path:
        """
        获取缓存文件路径

        Args:
            file_content: 文件二进制内容

        Returns:
            Path: 缓存文件完整路径
        """
        cache_key = self._generate_cache_key(file_content)
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_path: Path) -> Optional[List[TextNode]]:
        """
        从缓存加载文档（如果存在且有效）

        Args:
            cache_path: 缓存文件路径

        Returns:
            Optional[List[TextNode]]: 文档列表或None
        """
        if not cache_path.exists():
            singleton_logger.warning(f"缓存文件不存在: {cache_path}")
            return None

        try:
            with cache_path.open('r', encoding='utf-8') as f:
                nodes_data = json.load(f)
                # 从字典直接创建TextNode对象
                nodes = [TextNode.from_dict(node_dict)
                         for node_dict in nodes_data]
                singleton_logger.info(f"成功从缓存加载 {len(nodes)} 个文档块")
                return nodes
        except Exception as e:
            singleton_logger.warning(f"缓存加载失败（将重新生成）: {str(e)}")
            return None

    def _save_to_cache(self, cache_path: Path, nodes: List[TextNode]) -> bool:
        """
        保存处理结果到缓存

        Args:
            cache_path: 缓存文件路径
            nodes: 文档列表

        Returns:
            bool: 是否保存成功
        """
        try:
            # 转换为可序列化的字典格式
            serializable_nodes = [node.to_dict() for node in nodes]
            # 写入JSON文件
            with cache_path.open('w', encoding='utf-8') as f:
                json.dump(serializable_nodes, f, ensure_ascii=False, indent=4)
            singleton_logger.info(f"文档块缓存已保存: {cache_path}")
            return True
        except Exception as e:
            singleton_logger.error(f"缓存保存失败: {str(e)}")
            return False

    def _process_with_loader(
        self,
        file_content: bytes,
        file_extension: str
    ) -> List[TextNode]:
        """
        使用指定加载器处理文档

        处理流程：
        1. 检查缓存 -> 2. 创建临时文件 -> 3. 加载文档
        4. 分块处理 -> 5. 保存缓存 -> 6. 返回结果

        Args:
            file_content: 文件二进制内容
            file_extension: 文件扩展名（带点）

        Returns:
            List[TextNode]: 分块后的文档列表
        """
        # 获取缓存路径并检查缓存
        cache_path = self._get_cache_path(file_content)
        if cached_nodes := self._load_from_cache(cache_path):
            singleton_logger.info(f"缓存命中: {file_extension} 文档")
            return cached_nodes

        # 获取对应加载器
        loader_class = EXTENSION_READER_MAP.get(file_extension.lower())
        if not loader_class:
            singleton_logger.error(f"不支持的文件类型: {file_extension}")
            return []

        singleton_logger.info(
            f"处理 {file_extension} 文档 | 大小: {len(file_content)} 字节")

        try:
            # 创建临时文件（确保文件保持打开状态）
            with tempfile.NamedTemporaryFile(
                suffix=file_extension,
                mode='wb',
                delete=False  # 保持文件存在直到处理完成
            ) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            # 加载文档
            try:
                # 初始化文档加载器
                loader = loader_class()
                # 加载文档为LlamaIndex Document对象
                documents = loader.load_data(Path(tmp_file_path))

                # 分块处理（将文档转换为文档）
                nodes = self.text_splitter.get_nodes_from_documents(documents)

                # 保存结果到缓存
                if nodes:
                    self._save_to_cache(cache_path, nodes)
                return nodes
            finally:
                # 处理完成后删除临时文件
                Path(tmp_file_path).unlink(missing_ok=True)
        except Exception as e:
            singleton_logger.error(f"文档处理失败: {str(e)}")
            return []

    @error_handler()
    @log_execution
    def process_file(
        self,
        file_content: bytes,
        file_name: str
    ) -> List[TextNode]:
        """
        处理上传的文件（统一接口）

        Args:
            file_content: 文件二进制内容
            file_name: 完整文件名（带扩展名）

        Returns:
            List[TextNode]: 分块后的文档列表

        Raises:
            ValueError: 不支持的文件类型
        """
        # 提取文件扩展名
        file_extension = Path(file_name).suffix.lower()

        # 验证文件类型
        if file_extension not in EXTENSION_READER_MAP:
            supported_types = ", ".join(EXTENSION_READER_MAP.keys())
            singleton_logger.error(
                f"不支持的文件类型: {file_extension} | 支持类型: {supported_types}"
            )
            return []

        # 处理文档
        return self._process_with_loader(file_content, file_extension)
