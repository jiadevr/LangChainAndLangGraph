from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import pprint
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Initialize Tavily Search Tool
# Search负责搜索Extract负责抽取全部内容；建议先使用travily的API对比效果再确定工作流
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_domains=["dev.epicgames.com/documentation/unreal-engine/"]
)

# chatLLM = ChatOpenAI(
#     name="ollama-ai",
#     model="gemma4",
#     base_url="http://localhost:11434/v1",
#     api_key="123",
#     temperature=0.7,
# )

# llm_with_tools = chatLLM.bind_tools([tavily_search_tool])
# prompt=ChatPromptTemplate.from_messages([
#     ("system", "你是一个智能助手，你可以使用工具回答用户的问题"),
#     ("human", "{input}"),
# ])

# chain=prompt|llm_with_tools|StrOutputParser()

user_input = "LinearInterpolate AND Material"
response=tavily_search_tool.invoke({"query": user_input})
pprint.pprint(response)
