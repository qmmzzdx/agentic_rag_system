"""
天气查询工具 - 基于高德地图API实现天气查询功能
"""
import requests
from typing import Dict, Any, Optional, Tuple
from utils.logger.logger_manager import singleton_logger

# 导入配置项
from settings.system_settings import (
    AMAP_API_KEY,
    AMAP_WEATHER_API_URL,
    AMAP_GEO_API_URL
)


class WeatherService:
    """
    高德地图天气服务封装类

    功能：
    - 城市编码查询
    - 实时天气查询
    - 天气预报查询
    - 天气数据格式化
    """

    # 响应状态常量
    STATUS_SUCCESS = "success"
    STATUS_ERROR = "error"

    def __init__(self, api_key: str):
        """
        初始化天气服务

        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
        singleton_logger.info("天气服务初始化成功")

    def get_city_code(self, city_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        获取城市编码和标准化城市名称

        Args:
            city_name: 城市名称

        Returns:
            Tuple[Optional[str], Optional[str]]: (adcode, 标准城市名称) 
        """
        try:
            params = {
                "key": self.api_key,
                "address": city_name,
                "output": "JSON"
            }

            response = requests.get(
                AMAP_GEO_API_URL, params=params, timeout=10)
            response.raise_for_status()  # 检查HTTP状态
            data = response.json()

            if data.get("status") == "1" and int(data.get("count", 0)) > 0:
                geocode = data["geocodes"][0]
                city = geocode.get("city") or geocode.get(
                    "district") or city_name
                return geocode["adcode"], city
            else:
                singleton_logger.warning(f"未找到城市: {city_name}, 响应: {data}")
                return None, None

        except requests.exceptions.RequestException as e:
            singleton_logger.error(f"获取城市编码网络错误: {str(e)}")
        except Exception as e:
            singleton_logger.error(f"获取城市编码失败: {str(e)}", exc_info=True)
        return None, None

    def query_weather(self, city: str, extensions: str = "all") -> Dict[str, Any]:
        """
        查询天气信息

        Args:
            city: 城市名称或编码
            extensions: 气象类型，base-实况天气，all-预报天气（未来3天）

        Returns:
            Dict[str, Any]: 包含状态、数据和消息的字典
        """
        result = {
            "status": self.STATUS_ERROR,
            "data": None,
            "message": "",
            "summary": ""
        }

        try:
            # 获取城市编码
            city_code, city_name = self._get_city_info(city)
            if not city_code:
                result["message"] = f"无法找到城市: {city}"
                return result

            # 请求天气API
            weather_data = self._fetch_weather_data(city_code, extensions)
            if not weather_data:
                result["message"] = "天气数据获取失败"
                return result

            # 处理天气数据
            return self._process_weather_data(weather_data, city_name, extensions)

        except Exception as e:
            singleton_logger.error(f"查询天气时发生错误: {str(e)}", exc_info=True)
            result["message"] = f"查询天气时发生错误: {str(e)}"
            return result

    def _get_city_info(self, city: str) -> Tuple[Optional[str], Optional[str]]:
        """获取城市编码和名称"""
        if city.isdigit():
            # 如果输入的是城市编码，直接使用
            return city, city
        return self.get_city_code(city)

    def _fetch_weather_data(self, city_code: str, extensions: str) -> Optional[Dict]:
        """从API获取天气数据"""
        try:
            params = {
                "key": self.api_key,
                "city": city_code,
                "extensions": extensions,
                "output": "JSON"
            }

            response = requests.get(
                AMAP_WEATHER_API_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            singleton_logger.error(f"天气API请求失败: {str(e)}")
            return None

    def _process_weather_data(
        self,
        data: Dict,
        city_name: str,
        extensions: str
    ) -> Dict[str, Any]:
        """处理天气API响应数据"""
        result = {
            "status": self.STATUS_ERROR,
            "data": data,
            "message": "",
            "summary": ""
        }

        if data.get("status") != "1":
            result["message"] = f"天气查询失败，API返回: {data.get('info', '未知错误')}"
            return result

        result["status"] = self.STATUS_SUCCESS

        if extensions == "base":
            lives = data.get("lives", [])
            if lives:
                result["summary"] = self._format_current_weather(
                    lives[0], city_name)
            else:
                result["message"] = "未找到实时天气数据"
        else:
            forecasts = data.get("forecasts", [])
            if forecasts and forecasts[0].get("casts"):
                result["summary"] = self._format_forecast_weather(
                    forecasts[0], city_name)
            else:
                result["message"] = "未找到天气预报数据"

        return result

    def _format_current_weather(self, weather: Dict[str, Any], city_name: str) -> str:
        """格式化当前天气信息"""
        return (
            f"{city_name}当前天气: {weather.get('weather')}，"
            f"气温{weather.get('temperature')}℃，"
            f"湿度{weather.get('humidity')}%，"
            f"{weather.get('winddirection')}风{weather.get('windpower')}级。"
            f"更新时间: {weather.get('reporttime')}"
        )

    def _format_forecast_weather(self, forecast: Dict[str, Any], city_name: str) -> str:
        """格式化天气预报信息"""
        result = [f"{city_name}未来天气预报:"]

        for cast in forecast.get("casts", []):
            date = cast.get("date")
            day_weather = cast.get("dayweather")
            night_weather = cast.get("nightweather")
            day_temp = cast.get("daytemp")
            night_temp = cast.get("nighttemp")
            day_wind = f"{cast.get('daywind')}风{cast.get('daypower')}级"
            night_wind = f"{cast.get('nightwind')}风{cast.get('nightpower')}级"

            result.append(
                f"{date}: 白天{day_weather} {day_temp}℃ {day_wind}, "
                f"夜间{night_weather} {night_temp}℃ {night_wind}"
            )
        return "\n".join(result)


class WeatherTools:
    """
    天气查询工具类 - 提供简化的天气查询接口
    """

    def __init__(self, api_key: str = AMAP_API_KEY):
        """
        初始化天气工具

        Args:
            api_key: 高德地图API密钥，默认为配置中的密钥
        """
        self.weather_service = WeatherService(api_key)
        singleton_logger.info("天气查询工具初始化成功")

    def query_weather(self, city: str) -> str:
        """
        查询指定城市的天气预报

        Args:
            city: 要查询的城市名称

        Returns:
            str: 格式化的天气信息
        """
        try:
            # 首先尝试获取预报天气
            result = self.weather_service.query_weather(city, "all")

            if result["status"] == WeatherService.STATUS_SUCCESS and result["summary"]:
                return result["summary"]

            # 如果预报天气失败，尝试获取实时天气
            singleton_logger.info("预报天气获取失败，尝试获取实时天气")
            result = self.weather_service.query_weather(city, "base")

            if result["status"] == WeatherService.STATUS_SUCCESS and result["summary"]:
                return result["summary"]

            # 如果都失败，返回错误信息
            return f"获取{city}的天气信息失败: {result.get('message', '未知错误')}"

        except Exception as e:
            singleton_logger.error(f"查询天气时发生意外错误: {str(e)}", exc_info=True)
            return f"查询天气时发生系统错误: {str(e)}"
