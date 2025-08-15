from typing import Dict, Type
from llama_index.readers.file import FlatReader
from llama_index.readers.file import CSVReader
from llama_index.readers.file import MarkdownReader
from llama_index.readers.file import DocxReader
from .ocr_pdf_reader import OCRPDFReader

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
