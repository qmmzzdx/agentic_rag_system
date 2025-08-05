"""
工具配置 - 定义智能体可使用的各种工具
"""
from agent.tools.weather_tool import WeatherTools
from settings.system_settings import AMAP_API_KEY

# 天气查询工具
TOOL_WEATHER = {
    "name": "query_weather",
    "description": "查询指定城市的天气预报",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "要查询的城市名称"
            }
        },
        "required": ["city"]
    },
    "entrypoint": WeatherTools(AMAP_API_KEY).query_weather
}

# 可以继续添加其他工具...
# TOOL_CALCULATOR = {...}
