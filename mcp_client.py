from dotenv import load_dotenv
load_dotenv()
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys
from pathlib import Path
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()
from langchain_groq import ChatGroq
llm=ChatGroq(model="openai/gpt-oss-20b")
AVIATION_API_KEY=os.getenv("AVIATION_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY=os.getenv("OPENWEATHER_API_KEY")
# Resolve the absolute path to your custom server script
WEATHER_SERVER_PATH = Path(__file__).resolve().parent / "custom_mcp_server.py"

# 1. CRITICAL: Copy the current environment so the subprocess stays in the .venv!
WEATHER_ENV = os.environ.copy()

# 2. Add your custom environment variables to the copy
WEATHER_ENV.update({
    "WEATHER_API_KEY": os.getenv("WEATHER_API_KEY", ""),
    "PYTHONWARNINGS": "ignore"  # Always good for stdio MCP servers
})



client = MultiServerMCPClient({
    "tavily": {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
    },
  "aviationstack": {  
        "transport": "stdio",
        "command": "uvx",
        "args": [
            "--quiet",          # Silences uvx download logs
            "--with", "mcp<2",  # Forces uvx to use the correct MCP version
            "aviationstack-mcp"
        ],
        "env": {
            "AVIATION_STACK_API_KEY": AVIATION_API_KEY or "",
            "PYTHONWARNINGS": "ignore"  # <--- Hides the pydantic warning
        }
    },
     "weather": {
            "transport": "stdio",
            
            # Uses the .venv python exactly
            "command": sys.executable,
            
            "args": [
                str(WEATHER_SERVER_PATH)
            ],
            
            # Pass the merged environment
            "env": WEATHER_ENV
        }
})
async def get_all_tools():
    tools=await client.get_tools()
    print("\nAVALIABLE TOOL NAME\n")
    for tool in tools:
        print(tool.name)

aviation_tools={}
search_tool=None
async def initialize_mcp():
  global search_tool
  global aviation_tools
  if search_tool is not None and aviation_tools:
     return

  tools=await client.get_tools()
  print("\nAVALIABLE TOOLS\n")
  for tool in tools:
     print(tool.name)

  search_tool=next(
      tool
      for tool in tools
      if tool.name=="tavily_search"
   )
  aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name != "tavily_search"
    }

async def tavily_mcp_search(query:str):
   result=await search_tool.ainvoke({
      "query":query
   })
   return result

async def aviation_mcp_search(tool_name:str,tools_args:dict):
   tools=await client.get_tools()

   tool=next(
      t for t in tools
      if t.name==tool_name

   )
   result=await tool.ainvoke(
      tools_args or {}
   )
   return result

weather_tool=None
forecast_tool=None 
async def initialize_weather_tools():
 global weather_tool
 global forecast_tool
 if weather_tool is not None and forecast_tool is not None:
        return

    # 2. FIX: You MUST await the client call
 tools = await client.get_tools()
    
    # 3. FIX: Add 'None' as the default value to prevent crashes if a tool is missing
 weather_tool = next((t for t in tools if t.name == "get_city_weather"), None)
    
    # Note: Double-check if the tool is actually spelled "forcast_data" or "forecast_data"
 forecast_tool = next((t for t in tools if t.name == "forcast_data"), None)

async def weather_mcp_search(query:str):
 await initialize_weather_tools()
 return await weather_tool.ainvoke({
    "city":query
 })

async def forecast_mcp_search(query:str):
   await initialize_weather_tools()
   return await forecast_tool.ainvoke({
      "city":query
   })

def extract_destination(query: str):
    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()
