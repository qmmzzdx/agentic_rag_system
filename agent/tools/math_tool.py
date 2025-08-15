import re
import math
from typing import Any, List
from agno.tools import Toolkit
from utils.logger.logger_manager import singleton_logger


class MathTool(Toolkit):
    """
    安全计算数学表达式的工具集。支持加减乘除、指数、常用数学函数和常量。
    示例输入: "(2 + 3) * sin(pi/2) ^ 2"
    """

    def __init__(
        self,
        evaluate_expression: bool = True,
        describe_functions: bool = True,
        **kwargs
    ):
        """初始化数学工具集"""
        tools: List[Any] = []
        if evaluate_expression:
            tools.append(self.evaluate_expression)
        if describe_functions:
            tools.append(self.describe_functions)

        super().__init__(name="math_tool", tools=tools, **kwargs)

        # 安全函数和常量的白名单
        self.safe_functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'exp': math.exp, 'log': math.log, 'log10': math.log10,
            'sqrt': math.sqrt, 'abs': abs, 'round': round,
            'ceil': math.ceil, 'floor': math.floor
        }
        self.safe_constants = {
            'pi': math.pi,
            'e': math.e
        }
        singleton_logger.info("数学计算工具初始化完成")

    def evaluate_expression(self, expression: str) -> str:
        """计算数学表达式并返回格式化结果

        Args:
            expression: 要计算的数学表达式

        Returns:
            计算结果（若成功）或错误消息（若失败）
        """
        try:
            # 移除表达式中的额外空格
            cleaned_expr = re.sub(r'\s+', '', expression)

            # 安全检查：禁止双下划线（防止潜在风险）
            if '__' in cleaned_expr:
                return "表达式包含非法字符序列"

            # 执行计算
            result = self._safe_evaluate(cleaned_expr)

            # 格式化为最多保留6位小数
            return f"{expression} = {result:.6g}".rstrip('0').rstrip('.')
        except Exception as e:
            return f"计算错误: {str(e)}"

    def describe_functions(self) -> str:
        """列出所有支持的数学函数和常量

        Returns:
            支持的数学函数和常量的描述
        """
        func_list = "\n".join(
            [f"- {name}" for name in self.safe_functions.keys()])
        const_list = "\n".join(
            [f"- {name}: {value}" for name, value in self.safe_constants.items()])
        return f"支持的数学函数: {func_list}，支持的常量:{const_list}"

    def _safe_evaluate(self, expr: str) -> float:
        """安全地评估数学表达式，仅允许预定义的函数和常量"""
        # 使用正则表达式提取所有标识符（函数名和常量名）
        identifiers = set(re.findall(r'[a-zA-Z_]+', expr))

        # 检查所有标识符是否都是安全的
        unsafe_ids = [id for id in identifiers
                      if id not in self.safe_functions
                      and id not in self.safe_constants]
        if unsafe_ids:
            return f"检测到不安全标识符: {', '.join(unsafe_ids)}"

        # 创建安全的环境（仅包含预定义的函数和常量）
        safe_env = {**self.safe_functions, **self.safe_constants}

        # 替换表达式中的^为**（Python使用**表示指数）
        expr = expr.replace('^', '**')

        try:
            # 使用eval进行计算，但限制环境到安全函数/常量
            result = eval(expr, {"__builtins__": None}, safe_env)
            if not isinstance(result, (int, float)):
                return "结果不是数值类型"
            return result
        except Exception as e:
            return f"计算失败: {str(e)}"
