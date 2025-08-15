import io
import tqdm
import cv2
import fitz
import easyocr
from pathlib import Path
import numpy as np
from PIL import Image
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from typing import List, Optional, Dict
from utils.logger.logger_manager import singleton_logger


class OCRPDFReader(BaseReader):
    """
    使用OCR技术从PDF中提取文本

    处理包含扫描图像或混合内容的PDF文档，结合了两种技术：
    1. 原生文本提取：直接从PDF的可搜索文本层提取文本
    2. OCR处理：对PDF中的图像进行光学字符识别

    主要功能：
    - 处理扫描版PDF（无原生文本）
    - 处理混合内容PDF（部分文本+部分图像）
    - 支持多语言识别
    - 自动校正页面旋转
    - 智能过滤小图像提高效率
    """

    def __init__(self,
                 languages: List[str] = ['en', 'ch_sim'],  # 默认支持英文和简体中文
                 gpu: bool = False,                        # 是否使用GPU加速
                 pdf_ocr_threshold: tuple = (0.2, 0.2),    # 图像尺寸阈值
                 use_tqdm: bool = True):
        """
        初始化PDF OCR阅读器

        参数:
            languages: OCR支持的语言列表
                - 'en': 英语
                - 'ch_sim': 简体中文
                - 其他支持语言参考EasyOCR文档
            gpu: 是否使用GPU加速
                - True: 使用GPU（需要支持CUDA的NVIDIA显卡）
                - False: 使用CPU（默认）
            pdf_ocr_threshold: 图像尺寸阈值 (宽比例, 高比例)
                - 只处理宽度 > 页面宽度 * pdf_ocr_threshold[0] 且 
                  高度 > 页面高度 * pdf_ocr_threshold[1] 的图像
                - 避免处理小图标和装饰元素
            use_tqdm: 是否显示进度条
                - True: 显示处理进度（默认）
                - False: 不显示进度条
        """
        self.languages = languages
        self.gpu = gpu
        self.pdf_ocr_threshold = pdf_ocr_threshold
        self.use_tqdm = use_tqdm

        singleton_logger.info(
            f"正在初始化EasyOCR引擎，支持语言: {', '.join(self.languages)}...")
        self.reader = easyocr.Reader(
            lang_list=self.languages,      # 指定识别的语言
            gpu=self.gpu,                  # 是否使用GPU加速
            model_storage_directory=None,  # 模型存储目录（None表示使用默认位置）
            download_enabled=True          # 如果缺少模型，自动下载
        )
        singleton_logger.info(
            f"EasyOCR初始化完成！支持语言: {', '.join(self.languages)}")

    def _rotate_img(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像到正确方向（校正页面旋转）

        OCR算法对图像方向敏感，旋转后的文本识别率会下降。
        此函数确保图像以正确方向进行OCR处理。

        参数:
            img: 输入图像 (NumPy数组)
            angle: 旋转角度（度），正值表示逆时针旋转

        返回:
            旋转后的图像 (NumPy数组)

        算法步骤:
        1. 计算旋转中心（图像中心）
        2. 创建旋转矩阵
        3. 计算旋转后的新图像尺寸
        4. 调整旋转矩阵以保持图像完整
        5. 应用旋转变换
        """
        # 获取图像的高度和宽度
        h, w = img.shape[:2]
        # 计算图像旋转的中心点，这是通过将宽度和高度各自除以2来得到的。
        rotate_center = (w/2, h/2)
        # 获取旋转矩阵
        # 参数1为旋转中心点;
        # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
        # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算旋转后图像的新宽度和新高度
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        # 对旋转矩阵M进行调整，以考虑图像旋转后的平移
        # 目的是确保旋转后的图像能够完整显示，而不是部分被裁剪掉
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        # 使用cv2.warpAffine函数和旋转矩阵M对原图像进行变换，得到旋转后的图像rotated_img
        # 这里指定了变换后的新尺寸(new_w, new_h)
        return cv2.warpAffine(img, M, (new_w, new_h))

    def load_data(
        self,
        file_path: str,
        extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """
        从PDF文件加载文本内容（结合原生文本提取和OCR）

        处理流程:
        1. 打开PDF文档
        2. 遍历每一页
        3. 提取页面原生文本（如果有）
        4. 提取并处理页面中的图像
        5. 对每个符合条件的图像进行OCR
        6. 合并原生文本和OCR结果
        7. 为每页创建一个Document对象

        参数:
            file_path: PDF文件路径
            extra_info: 添加到文档元数据的额外信息（可选）

        返回:
            Document对象列表（每页一个文档）
        """
        # 存储所有页面文档
        documents = []

        # 使用PyMuPDF提取文本和图像
        pdf_doc = fitz.open(file_path)
        total_pages = pdf_doc.page_count
        singleton_logger.info(f"正在处理PDF: {file_path}，共 {total_pages} 页")

        # 配置进度条
        pbar = tqdm.tqdm(
            total=total_pages,
            desc="处理PDF页面",
            disable=not self.use_tqdm  # 根据参数决定是否显示进度条
        )

        # 遍历PDF的每一页
        for page_num in range(total_pages):
            # 更新进度条
            pbar.update(1)
            pbar.set_description(f"处理第 {page_num+1}/{total_pages} 页")

            try:
                # 加载当前页
                page = pdf_doc.load_page(page_num)
                page_text = ""  # 存储当前页的文本内容

                # 提取原生文本（如果PDF是可搜索的）
                # 使用"text"参数获取纯文本，保留基本格式
                text = page.get_text("text").strip()
                if text:
                    page_text += text + "\n\n"  # 添加换行分隔符

                # 处理页面中的图像
                # get_images(full=True)返回页面中的所有图像信息
                img_list = page.get_images(full=True)

                # 遍历页面中的每张图像
                for img_index, img_info in enumerate(img_list):
                    # 图像信息中的第一个元素是交叉引用号(xref)
                    xref = img_info[0]

                    try:
                        # 使用xref提取图像数据
                        base_image = pdf_doc.extract_image(xref)

                        # 获取图像的二进制数据
                        img_bytes = base_image["image"]

                        # 使用PIL从字节数据创建图像对象
                        # 这样可以处理多种图像格式
                        img = Image.open(io.BytesIO(img_bytes))

                        # 转换图像模式为RGB（如果需要）
                        # OCR通常需要RGB格式，RGBA（带透明度）需要转换
                        if img.mode == 'RGBA':
                            # 创建一个白色背景，并将RGBA图像合成到上面
                            # 这样处理可以避免透明部分影响OCR
                            background = Image.new(
                                'RGB', img.size, (255, 255, 255))
                            # 使用alpha通道作为掩码
                            background.paste(img, mask=img.split()[3])
                            img = background
                        elif img.mode != 'RGB':
                            # 转换为RGB格式
                            img = img.convert('RGB')

                        # 将PIL图像转换为NumPy数组（OpenCV需要）
                        img_array = np.array(img)

                        # 检查图像尺寸是否超过阈值（避免处理小图标）
                        # 计算图像相对于页面的宽度和高度比例
                        img_width_ratio = img_array.shape[1] / page.rect.width
                        img_height_ratio = img_array.shape[0] / \
                            page.rect.height

                        # 如果图像太小，跳过OCR处理
                        if (img_width_ratio < self.pdf_ocr_threshold[0] or
                                img_height_ratio < self.pdf_ocr_threshold[1]):
                            continue

                        # 处理页面旋转（如果有）
                        # PDF页面可能有旋转设置，需要校正图像方向
                        rotation = page.rotation
                        if rotation != 0:
                            # 转换为OpenCV格式 (BGR)
                            # OpenCV默认使用BGR格式，而PIL是RGB
                            img_array_bgr = cv2.cvtColor(
                                img_array, cv2.COLOR_RGB2BGR)

                            # 旋转图像（校正方向）
                            # 360-rotation 是因为PDF的旋转是顺时针定义，而我们的函数是逆时针
                            rotated_img = self._rotate_img(
                                img_array_bgr, 360 - rotation)

                            # 转换回RGB格式（EasyOCR需要）
                            img_array = cv2.cvtColor(
                                rotated_img, cv2.COLOR_BGR2RGB)
                        else:
                            # 不需要旋转，保持原样
                            img_array = img_array

                        # 执行EasyOCR光学字符识别
                        # EasyOCR是基于深度学习的OCR引擎，包含以下步骤：
                        # a. 文本检测：定位图像中的文本区域
                        # b. 文本识别：识别文本区域中的字符
                        try:
                            # 调用EasyOCR识别图像中的文本
                            # reader.readtext()返回检测结果列表
                            # detail=0 表示只返回文本内容，不返回位置和置信度
                            results = self.reader.readtext(img_array, detail=0)
                            if results:
                                # 合并所有识别结果
                                ocr_text = "\n".join(results)
                                # 将OCR结果添加到页面文本
                                # 添加标记以便区分OCR文本和原生文本
                                page_text += f"\n[图像 {img_index+1} OCR结果]:\n{ocr_text}\n\n"
                        except Exception as e:
                            # OCR处理失败，打印错误但继续处理其他图像
                            singleton_logger.error(
                                f"第{page_num+1}页图像{img_index+1}OCR处理失败: {str(e)}")
                    except Exception as e:
                        # 图像处理失败，打印错误但继续处理其他图像
                        singleton_logger.error(
                            f"第{page_num+1}页图像{img_index+1}处理失败: {str(e)}")

                # 创建当前页的Document对象
                file_path_str = str(Path(file_path)) if isinstance(
                    file_path, Path) else file_path
                metadata = {
                    "source": file_path_str,       # 文件路径
                    "page": page_num + 1,          # 页码（从1开始）
                    "total_pages": total_pages,    # 总页数
                    "native_text": bool(text),     # 是否有原生文本
                    "image_count": len(img_list)   # 图像数量
                }

                # 添加额外元数据（如果有）
                if extra_info:
                    metadata.update(extra_info)
                # 创建Document对象，包含当前页的文本和元数据
                documents.append(
                    Document(text=page_text.strip(), metadata=metadata))

            except Exception as e:
                # 页面处理失败，创建错误文档
                singleton_logger.error(f"处理第 {page_num+1} 页时出错: {str(e)}")
                documents.append(Document(
                    text=f"[错误] 无法处理第{page_num+1}页: {str(e)}",
                    metadata={
                        "source": file_path,
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "error": str(e)
                    }
                ))

        # 关闭进度条和PDF文档
        pbar.close()
        pdf_doc.close()
        singleton_logger.info(f"PDF处理完成，共提取 {len(documents)} 页内容")
        return documents
