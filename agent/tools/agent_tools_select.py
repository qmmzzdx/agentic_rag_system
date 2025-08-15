# 导入数学计算工具类
from .math_tool import MathTool
# 导入天气查询工具类
from .weather_tool import WeatherTool

# 工具列表，包含所有可用的工具实例
# 用于在agent中统一管理和调用各种工具
TOOL_LISTS = [
    MathTool(),      # 数学计算工具实例
    WeatherTool()    # 天气查询工具实例
]
