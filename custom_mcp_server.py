from mcp.server.fastmcp import FastMCP
import requests 
import os
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("Weather MCP Server")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@mcp.tool()
def get_city_weather(city: str):
    response = requests.get(
        url="https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    )
    
    # FIX: Added parentheses to actually call the json function
    data = response.json() 
    
    if response.status_code != 200:
        return data
        
    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"], # FIX: Added 's'
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"], # FIX: Changed "main" to "weather"
        "wind_speed": data["wind"]["speed"],
    }

# FIX: Corrected decorator from @mcp_tool() to @mcp.tool()
@mcp.tool()
def forcast_data(city: str):
    # FIX: Changed endpoint from /weather to /forecast
    response = requests.get(
        url="https://api.openweathermap.org/data/2.5/forecast",
        params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    )
    
    data = response.json()
    
    if response.status_code != 200:
        return data
        
    forecast = []
    for item in data.get("list", [])[:5]:
        forecast.append({
            "datetime": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "weather": item["weather"][0]["description"]
        })

    return {
       "city": city,
       "forecast": forecast
    }

if __name__ == "__main__":
    # Starts the standard I/O server LangChain is expecting
    mcp.run()