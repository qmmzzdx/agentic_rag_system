"""
日志管理类 - 提供全局单例日志记录器，支持控制台和文件输出

功能特性：
- 全局单例模式，确保整个应用使用同一个日志实例
- 双输出渠道：带颜色的控制台输出和持久化文件输出
- 智能日志管理：每日单个日志文件，大小超过20MB自动截断
- 完善的错误处理：自动捕获日志处理过程中的异常
- 丰富的日志级别：支持从TRACE到CRITICAL的多级别日志记录
"""
from pathlib import Path
from loguru import logger


class SingleTonLogger:
    """
    单例日志管理器类，负责初始化日志配置并提供全局访问接口

    使用示例：
        from utils.logger import singleton_logger as log
        log.info("应用启动")
        log.error("操作失败", exc_info=True)
    """

    def __init__(self):
        """初始化日志管理器，创建日志目录并配置日志处理器"""
        # 创建日志目录（如果不存在）
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # 配置日志处理器
        self._configure_logger()

    def _configure_logger(self):
        """
        配置日志记录器处理器

        配置说明：
        1. 控制台输出：
           - 彩色格式化输出
           - 仅记录INFO及以上级别日志
           - 格式：时间 | 日志级别 | 模块:行号 - 消息

        2. 文件输出：
           - 每日生成单个日志文件（ars-YYYY-MM-DD.log）
           - 文件大小超过20MB自动创建新文件
           - 保留当天日志文件
           - 记录DEBUG及以上级别日志
           - 包含异常堆栈和变量诊断信息
        """
        # 移除所有默认处理器
        logger.remove()

        # 配置控制台输出处理器
        logger.add(
            sink=lambda msg: print(msg, end=""),  # 直接打印到控制台
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{module}:{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )

        # 配置文件输出处理器
        logger.add(
            sink=self.log_dir / "ars-{time:YYYY-MM-DD}.log",
            rotation="20 MB",    # 文件大小达到20MB时轮转
            retention="1 day",   # 仅保留当天日志文件
            compression=None,    # 不压缩日志文件
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{line} - {message}",
            level="DEBUG",       # 记录DEBUG及以上级别日志
            enqueue=True,        # 启用线程安全
            backtrace=True,      # 记录异常堆栈
            diagnose=True,       # 记录变量诊断信息
            catch=True           # 捕获日志处理过程中的异常
        )

    def get_logger(self):
        """
        获取配置好的日志记录器实例

        Returns:
            Logger: 配置完成的Loguru日志记录器实例
        """
        return logger


# 创建全局单例日志记录器
singleton_logger = SingleTonLogger().get_logger()
