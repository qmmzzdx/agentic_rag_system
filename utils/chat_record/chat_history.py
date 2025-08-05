"""
对话历史管理类 - 提供按日期分组的对话历史存储、加载和导出功能
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
from utils.logger.logger_manager import singleton_logger


class ChatHistoryManager:
    """
    对话历史管理器类，按日期组织对话历史文件

    功能特点：
    - 自动创建chat_history目录存储历史记录
    - 每天生成单独的JSON历史文件（格式：ars-chat-history-YYYYMMDD.json）
    - 提供历史记录的增删改查功能
    - 支持历史记录格式化和统计
    """

    def __init__(self):
        """初始化对话历史管理器，创建存储目录"""
        # 创建历史记录存储目录
        self.history_dir = Path("chat_history")
        self.history_dir.mkdir(exist_ok=True)

        # 初始化当前日期的历史文件路径
        self.current_history_file = self._get_today_history_file()
        self.history: List[Dict] = self.load_history()

    def _get_today_history_file(self) -> Path:
        """
        获取当天历史记录文件路径

        Returns:
            Path: 当天历史记录文件路径（格式：chat_history/ars-chat-history-YYYYMMDD.json）
        """
        today_str = datetime.now().strftime("%Y%m%d")
        return self.history_dir / f"ars-chat-history-{today_str}.json"

    def load_history(self) -> List[Dict]:
        """
        从当天历史文件加载对话历史

        Returns:
            List[Dict]: 包含历史消息的字典列表，每个字典包含'role'和'content'键
                        如果文件不存在或加载失败则返回空列表
        """
        try:
            if self.current_history_file.exists():
                with open(self.current_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                singleton_logger.warning(
                    f"历史文件不存在: {self.current_history_file}")
                return []
        except Exception as e:
            singleton_logger.error(
                f"加载历史记录失败[{self.current_history_file}]: {str(e)}")
        return []

    def save_history(self) -> None:
        """将当前对话历史保存到当天历史文件中"""
        try:
            with open(self.current_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=4)
        except Exception as e:
            singleton_logger.error(
                f"保存历史记录失败[{self.current_history_file}]: {str(e)}")

    def add_message(self, role: str, content: str) -> None:
        """
        添加新消息到历史记录并自动保存

        Args:
            role (str): 消息角色 ('user' 或 'assistant')
            content (str): 消息文本内容
        """
        # 检查是否需要切换到新日期的文件
        new_file = self._get_today_history_file()
        if new_file != self.current_history_file:
            # 保存当前日期的历史记录
            self.save_history()

            # 切换到新日期的文件
            self.current_history_file = new_file
            self.history = self.load_history()  # 加载新日期的历史记录

        self.history.append({"role": role, "content": content})
        self.save_history()

    def clear_history(self) -> None:
        """清空内存中的历史记录并删除当天历史文件"""
        self.history = []
        try:
            if self.current_history_file.exists():
                os.remove(self.current_history_file)
        except Exception as e:
            singleton_logger.error(
                f"删除历史文件失败[{self.current_history_file}]: {str(e)}")

    def export_to_csv(self, date_str: Optional[str] = None) -> Optional[bytes]:
        """
        导出指定日期的历史记录为CSV

        Args:
            date_str (str, optional): 日期字符串(YYYYMMDD格式)，None表示当天

        Returns:
            Optional[bytes]: CSV文件内容(UTF-8编码)，失败返回None
        """
        try:
            target_file = self._get_history_file_by_date(
                date_str) if date_str else self.current_history_file
            if not target_file.exists():
                return None

            with open(target_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            df = pd.DataFrame(history)
            return df.to_csv(index=False).encode('utf-8')
        except Exception as e:
            singleton_logger.error(f"导出历史记录失败[{date_str}]: {str(e)}")
            return None

    def _get_history_file_by_date(self, date_str: str) -> Path:
        """
        根据日期字符串获取历史文件路径

        Args:
            date_str (str): 日期字符串(YYYYMMDD格式)

        Returns:
            Path: 对应日期的历史文件路径
        """
        return self.history_dir / f"ars-chat-history-{date_str}.json"

    def get_stats(self) -> Dict[str, int]:
        """
        获取当天对话历史的统计信息

        Returns:
            Dict[str, int]: 包含total_messages和user_messages的字典
        """
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_messages": len(self.history),
            "user_messages": sum(1 for msg in self.history if msg["role"] == "user")
        }
