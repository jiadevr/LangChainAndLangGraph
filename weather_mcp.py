# [Model Context Protocol (MCP) - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/mcp)
# [万维易源-实时未来历史天气数据查询服务-云市场-阿里云](https://market.aliyun.com/detail/cmapi033617)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
import os
from langchain.agents import create_agent
import asyncio

async def main():
    chat_llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        temperature=0.7,
        timeout=10,
        max_completion_tokens=1000,
    )

    # 百链提供的MCP配置示例
    # {
    #   "mcpServers": {
    #     "mcp-Y2U3ZTc3MzM0NzJk": {
    #       "httpUrl": "https://dashscope.aliyuncs.com/api/v1/mcps/mcp-Y2U3ZTc3MzM0NzJk/mcp",
    #       "headers": {
    #         "Authorization": "Bearer ${DASHSCOPE_API_KEY}"
    #       }
    #     }
    #   }
    # }
    mcp_client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "https://dashscope.aliyuncs.com/api/v1/mcps/mcp-Y2U3ZTc3MzM0NzJk/mcp",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
                }
            }
        }
    )
    tools=await mcp_client.get_tools()
    
    agent = create_agent(
        model=chat_llm,
        tools=tools
    )
    weather_result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "查询北京天气"}]
    })
    print(weather_result)
    
if __name__ == "__main__":
    asyncio.run(main())
