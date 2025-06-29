"""
对话历史管理类 - 提供对话历史的存储、加载、格式化和导出功能
"""
import os
import json
import logging
from typing import List, Dict, Optional
import pandas as pd
from config.settings import HISTORY_FILE, MAX_HISTORY_TURNS  # 导入配置文件中的路径和最大历史轮数设置

# 配置日志记录器
logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """
    对话历史管理器类，负责处理对话历史的存储、检索和转换操作
    """

    def __init__(self):
        """初始化对话历史管理器，自动加载已有的对话历史"""
        self.history: List[Dict] = self.load_history()

    # 1. 从文件加载对话历史
    def load_history(self) -> List[Dict]:
        """
        从指定文件路径加载JSON格式的对话历史

        Returns:
            List[Dict]: 包含历史消息的字典列表，每个字典包含'role'和'content'键
                        如果加载失败则返回空列表
        """
        try:
            # 检查历史文件是否存在
            if os.path.exists(HISTORY_FILE):
                # 以UTF-8编码打开并读取JSON文件
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            # 记录加载错误日志
            logger.error(f"加载对话历史时出错: {str(e)}")
        return []

    # 2. 保存对话历史到文件
    def save_history(self) -> None:
        """将当前对话历史保存到JSON文件中"""
        try:
            # 以缩进为4的格式写入JSON文件
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=4)
        except Exception as e:
            # 记录保存错误日志
            logger.error(f"保存对话历史时出错: {str(e)}")

    # 3. 添加新消息到历史记录
    def add_message(self, role: str, content: str) -> None:
        """
        添加新消息到历史记录并自动保存

        Args:
            role (str): 消息角色 ('user' 或 'assistant')
            content (str): 消息文本内容
        """
        # 将新消息添加到历史记录列表
        self.history.append({"role": role, "content": content})
        # 保存更新后的历史记录
        self.save_history()

    # 4. 清空对话历史
    def clear_history(self) -> None:
        """清空内存中的历史记录并删除历史文件"""
        # 清空内存中的历史记录
        self.history = []
        # 如果历史文件存在，则删除
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)

    # 5. 获取格式化的对话历史
    def get_formatted_history(self, max_turns: int = MAX_HISTORY_TURNS) -> str:
        """
        将最近的对话历史格式化为易读的字符串

        Args:
            max_turns (int): 最大保留的对话轮数（每轮包含用户+助手消息）

        Returns:
            str: 格式化后的对话历史字符串，包含角色标识和消息内容
        """
        # 如果没有历史记录，返回空字符串
        if not self.history:
            return ""

        # 计算需要保留的消息数量（每轮2条消息）
        max_messages = max_turns * 2
        # 获取最近的N条消息（如果总消息数不足则获取全部）
        recent_history = self.history[-max_messages:] if len(
            self.history) > max_messages else self.history

        # 构建格式化字符串
        formatted_history = "以下是之前的对话历史：\n"
        for msg in recent_history:
            # 将角色标识转换为中文
            role = "用户" if msg["role"] == "user" else "助手"
            # 添加消息到结果字符串
            formatted_history += f"{role}: {msg['content']}\n"

        return formatted_history

    # 6. 导出对话历史为CSV文件
    def export_to_csv(self) -> Optional[bytes]:
        """
        将对话历史导出为CSV格式的字节数据

        Returns:
            Optional[bytes]: UTF-8编码的CSV文件内容，失败时返回None
        """
        try:
            # 将历史记录转换为Pandas DataFrame
            df = pd.DataFrame(self.history)
            # 将DataFrame转换为CSV格式的字节数据（不带索引）
            return df.to_csv(index=False).encode('utf-8')
        except Exception as e:
            # 记录导出错误日志
            logger.error(f"导出对话历史时出错: {str(e)}")
            return None

    # 7. 获取对话历史统计信息
    def get_stats(self) -> Dict[str, int]:
        """
        获取对话历史的统计信息

        Returns:
            Dict[str, int]: 包含统计信息的字典：
                - total_messages: 总消息数
                - user_messages: 用户消息数
        """
        return {
            "total_messages": len(self.history),
            "user_messages": sum(1 for msg in self.history if msg["role"] == "user")
        }
