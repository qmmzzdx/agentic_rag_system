"""
智能体模型类 - 封装RAG智能体的核心功能
"""
from typing import Optional, List
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.reasoning import ReasoningTools
from agno.tools.function import Function
from config.settings import DEFAULT_MODEL
import logging
import importlib
import sys

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG智能体类，封装了与模型交互的功能

    特点：
    - 工具配置外部化，支持动态加载
    - 指令模板可配置
    - 支持多种工作模式（RAG/普通对话）
    - 修复了动态导入时的文件监视问题
    """

    def __init__(
        self,
        model_version: str = DEFAULT_MODEL,
        tool_config: str = "config.agent_tools",
        instruction_template: str = "config.agent_instructions"
    ):
        """
        初始化RAG智能体

        Args:
            model_version (str): 模型版本名称
            tool_config (str): 工具配置模块路径
            instruction_template (str): 指令模板模块路径
        """
        self.model_version = model_version
        self.tool_config = tool_config
        self.instruction_template = instruction_template
        self.agent = self._create_agent()

    def _safe_import(self, module_path: str) -> object:
        """
        安全导入模块，避免生成.pyc缓存文件

        Args:
            module_path: 模块路径

        Returns:
            导入的模块对象
        """
        # 禁用字节码生成
        sys.dont_write_bytecode = True

        try:
            module = importlib.import_module(module_path)
            return module
        except ImportError as e:
            logger.error(f"导入模块 {module_path} 失败: {str(e)}")
            return None
        finally:
            # 恢复字节码生成设置
            sys.dont_write_bytecode = False

    def _load_tools(self) -> List[Function]:
        """
        动态加载工具配置

        Returns:
            List[Function]: 配置好的工具列表
        """
        tools_module = self._safe_import(self.tool_config)
        if not tools_module:
            logger.error(f"工具配置模块 {self.tool_config} 加载失败，使用默认工具")
            return [ReasoningTools(add_instructions=True)]

        tools = []
        # 加载所有以TOOL_开头的配置项
        for attr_name in dir(tools_module):
            if attr_name.startswith("TOOL_"):
                tool_config = getattr(tools_module, attr_name)

                # 创建Function实例
                tool = Function(
                    name=tool_config["name"],
                    description=tool_config["description"],
                    parameters=tool_config.get("parameters", {}),
                    entrypoint=tool_config["entrypoint"]
                )
                tools.append(tool)

        # 添加默认的推理工具
        tools.append(ReasoningTools(add_instructions=True))
        logger.info(f"已加载 {len(tools)} 个工具")
        return tools

    def _load_instructions(self) -> str:
        """
        加载指令模板

        Returns:
            str: 格式化后的指令文本
        """
        prompts_module = self._safe_import(self.instruction_template)
        if not prompts_module:
            logger.error(f"指令模板模块 {self.instruction_template} 加载失败，使用默认指令")
            return """你是一个智能助手，可以回答用户的各种问题。
                    请确保对接收的内容以及需要做出的判断进行思考，
                    回答要简明、准确、有帮助。"""

        return getattr(prompts_module, "AGENT_INSTRUCTIONS", "")

    def _get_template(self, module: object, attr_name: str, default: str) -> str:
        """
        安全获取模板属性

        Args:
            module: 模块对象
            attr_name: 属性名称
            default: 默认模板

        Returns:
            模板字符串
        """
        try:
            return getattr(module, attr_name, default)
        except Exception as e:
            logger.error(f"获取模板 {attr_name} 失败: {str(e)}")
            return default

    def _create_agent(self) -> Agent:
        """
        创建Agent实例

        Returns:
            Agent: 配置好的Agent实例
        """
        # 加载工具和指令
        tools = self._load_tools()
        instructions = self._load_instructions()

        return Agent(
            name=f"{self.model_version} RAG Agent",
            model=Ollama(id=self.model_version),
            instructions=instructions,
            tools=tools,
            show_tool_calls=True,
            markdown=True,
        )

    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """
        构建最终提示词，根据有无上下文选择不同模板

        Args:
            prompt (str): 用户原始提示
            context (Optional[str]): 检索到的上下文

        Returns:
            str: 格式化后的完整提示
        """
        prompts_module = self._safe_import(self.instruction_template)
        if not prompts_module:
            logger.error("提示模板加载失败，使用默认提示")
            return f"【用户问题】\n{prompt}"

        if context:
            # RAG模式模板
            template = self._get_template(
                prompts_module,
                "RAG_PROMPT_TEMPLATE",
                "【检索内容】\n{context}\n\n【用户问题】\n{prompt}"
            )
            return template.format(context=context, prompt=prompt)
        else:
            # 普通对话模式模板
            template = self._get_template(
                prompts_module,
                "STANDARD_PROMPT_TEMPLATE",
                "【用户问题】\n{prompt}"
            )
            return template.format(prompt=prompt)

    def run(self, prompt: str, context: Optional[str] = None) -> str:
        """
        运行智能体处理查询

        Args:
            prompt (str): 用户输入的提示
            context (Optional[str]): 可选的文档上下文

        Returns:
            str: 智能体的响应
        """
        full_prompt = self._build_prompt(prompt, context)
        try:
            response = self.agent.run(full_prompt)
            return response.content
        except Exception as e:
            logger.error(f"智能体执行失败: {str(e)}")
            return "抱歉，处理您的请求时出现问题，请稍后再试。"
