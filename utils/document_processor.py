"""
文档处理模块 - 提供多种文档类型的加载、处理和缓存功能

功能：
- PDF文档解析与分块处理
- 处理结果缓存机制
- 多线程处理支持
- 多种文档类型支持（PDF/TXT）
"""
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Union
from utils.decorators import error_handler, log_execution

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 导入配置项
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

# 配置日志
logger = logging.getLogger(__name__)

# 支持的文件类型
SUPPORTED_EXTENSIONS = {
    '.pdf': PyPDFLoader,  # PDF文件处理类
    '.txt': TextLoader,   # 文本文件处理类
    # 可扩展更多文件类型
}


class DocumentProcessor:
    """
    文档处理器类，提供文档加载、处理和缓存功能

    特性：
    - 基于文件内容的智能缓存
    - 多线程处理
    - 自动清理临时文件
    - 可扩展的文件类型支持
    """

    def __init__(self, cache_dir: str = ".document_cache", max_workers: int = 4):
        """
        初始化文档处理器

        Args:
            cache_dir: 缓存目录路径，用于存储处理过的文档
            max_workers: 最大工作线程数，用于并行处理文档
        """
        # 创建缓存目录
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # 初始化文本分割器，用于将大文档分割成小块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,         # 每个文本块的大小
            chunk_overlap=CHUNK_OVERLAP,   # 块之间的重叠大小
            separators=SEPARATORS,         # 用于分割文本的分隔符
            length_function=len,           # 计算文本长度的函数
            is_separator_regex=False       # 分隔符是否为正则表达式
        )
        logger.info(f"文档处理器初始化完成，缓存目录: {self.cache_dir}")

    def _generate_cache_key(self, file_content: bytes, file_name: str) -> str:
        """
        生成唯一的缓存键（基于文件内容和文件名）

        Args:
            file_content: 文件的二进制内容
            file_name: 文件名

        Returns:
            str: 唯一的缓存键（MD5哈希值）
        """
        # 结合文件内容和文件名生成唯一哈希值
        return hashlib.md5(file_content + file_name.encode()).hexdigest()

    def _get_cache_path(self, file_content: bytes, file_name: str) -> Path:
        """
        获取缓存文件路径

        Args:
            file_content: 文件的二进制内容
            file_name: 文件名

        Returns:
            Path: 缓存文件的完整路径
        """
        # 基于缓存键生成缓存文件路径
        cache_key = self._generate_cache_key(file_content, file_name)
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_path: Path) -> Optional[List[Document]]:
        """
        从缓存加载处理结果

        Args:
            cache_path: 缓存文件路径

        Returns:
            Optional[List[Document]]: 缓存的文档列表，如果缓存不存在或无效则返回None
        """
        try:
            # 检查缓存文件是否存在
            if cache_path.exists():
                # 读取缓存文件内容
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 将JSON数据转换为Document对象列表
                    if isinstance(data, list):
                        return [Document(**doc) for doc in data]
        except (json.JSONDecodeError, OSError) as e:
            # 缓存加载失败时的警告（不影响正常流程）
            logger.warning(f"缓存加载失败（已忽略）: {cache_path} - {str(e)}")
        except Exception as e:
            # 其他异常记录错误日志
            logger.error(f"缓存加载异常: {str(e)}", exc_info=True)
        return None

    def _save_to_cache(self, cache_path: Path, documents: List[Document]) -> bool:
        """
        保存处理结果到缓存

        Args:
            cache_path: 缓存文件路径
            documents: 要缓存的文档列表

        Returns:
            bool: 是否保存成功
        """
        try:
            # 将Document对象转换为可序列化的字典
            docs_data = [doc.model_dump() for doc in documents]
            # 写入缓存文件
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, TypeError) as e:
            # 文件操作或类型转换错误
            logger.error(f"缓存保存失败: {str(e)}")
        except Exception as e:
            # 其他异常记录错误日志
            logger.error(f"缓存保存异常: {str(e)}", exc_info=True)
        return False

    @error_handler()
    @log_execution
    def _process_pdf(self, file_content: bytes, file_name: str) -> List[Document]:
        """
        处理PDF文件：加载、分割和缓存

        Args:
            file_content: PDF文件的二进制内容
            file_name: PDF文件名

        Returns:
            List[Document]: 处理后的文档块列表
        """
        # 检查缓存是否存在
        cache_path = self._get_cache_path(file_content, file_name)
        cached_docs = self._load_from_cache(cache_path)
        if cached_docs is not None:
            logger.info(f"从缓存加载文件: {file_name}")
            return cached_docs

        # 处理PDF文件（缓存未命中）
        logger.info(f"处理文件: {file_name}")

        try:
            # 创建临时文件（自动清理）
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
                # 写入文件内容
                temp_file.write(file_content)
                temp_file.flush()  # 确保内容写入磁盘

                # 使用PyPDFLoader加载PDF文档
                loader = PyPDFLoader(temp_file.name)
                documents = loader.load()

                # 使用文本分割器分割文档
                split_docs = self.text_splitter.split_documents(documents)

                # 保存处理结果到缓存
                if split_docs:
                    self._save_to_cache(cache_path, split_docs)

                return split_docs

        except Exception as e:
            logger.error(f"处理PDF文件失败: {str(e)}")
            raise

    def clear_cache(self):
        """清除所有缓存文件"""
        try:
            # 遍历缓存目录中的所有JSON文件
            for file in self.cache_dir.glob("*.json"):
                file.unlink()  # 删除文件
            logger.info("缓存已清除")
        except Exception as e:
            logger.error(f"清除缓存失败: {str(e)}")
            raise

    @error_handler()
    @log_execution
    def process_file(self, uploaded_file_or_content, file_name: str = None) -> Union[str, List[Document]]:
        """
        处理上传的文件，支持多种文件类型

        Args:
            uploaded_file_or_content: Streamlit上传的文件对象或文件内容
            file_name: 文件名（当第一个参数是文件内容时需要提供）

        Returns:
            Union[str, List[Document]]: 
                - PDF文件：返回文本内容字符串（Streamlit上传时）或文档块列表
                - TXT文件：返回文本内容字符串
                - 其他文件类型：返回错误信息字符串
        """
        try:
            # 判断输入类型
            if hasattr(uploaded_file_or_content, 'getvalue') and hasattr(uploaded_file_or_content, 'name'):
                # Streamlit上传的文件对象
                file_content = uploaded_file_or_content.getvalue()
                file_name = uploaded_file_or_content.name
            elif isinstance(uploaded_file_or_content, bytes) and file_name:
                # 直接传入的文件内容和文件名
                file_content = uploaded_file_or_content
            else:
                raise ValueError("参数错误：需要提供有效的文件对象或文件内容和文件名")

            # 根据文件扩展名进行特定处理
            if file_name.lower().endswith('.pdf'):
                # 处理PDF文件
                docs = self._process_pdf(file_content, file_name)

                # 返回格式处理：
                # - Streamlit上传时返回拼接的文本内容
                # - 其他情况返回文档块列表
                if hasattr(uploaded_file_or_content, 'getvalue'):
                    return "\n\n".join(doc.page_content for doc in docs)
                return docs

            elif file_name.lower().endswith('.txt'):
                # 处理TXT文件：直接返回解码后的文本内容
                return file_content.decode('utf-8')

            else:
                # 不支持的文件类型
                return f"不支持的文件类型: {file_name}"

        except Exception as e:
            logger.error(f"处理文件失败: {str(e)}")
            raise Exception(f"处理文件失败: {str(e)}")
