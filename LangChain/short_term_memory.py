# 短期记忆管理[Short-term memory - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/short-term-memory)
# 使用AgentState保存运行时信息，使用summarization_middleware中间件压缩上下文
from langchain.agents.middleware import SummarizationMiddleware,wrap_tool_call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from textwrap import dedent
import os

from langgraph.types import Command

chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)

# 摘要模型 使用qwen flash
summary_llm = chat_llm.model_copy(update={"model": "qwen-flash"})

summarization_middleware = SummarizationMiddleware(
    model=summary_llm, trigger=("tokens", 1000), keep=("messages", 2)
)


class CustomAgentState(AgentState):
    user_name: str
    user_location: str


@tool
def greet(
    user_name: str, user_location: str, runtime: ToolRuntime[CustomAgentState]
) -> str | Command:
    """
    当用户第一次发给你基本信息时调用这个函数并和用户打招呼
    Parameters:
    - user_name: 用户姓名
    - user_location: 用户位置
    """
    return Command(
        update={
            "user_name": user_name,
            "user_location": user_location,
            "messages": [
                ToolMessage("已更新用户信息", tool_call_id=runtime.tool_call_id)
            ],
        }
    )

@tool
def get_weather(runtime: ToolRuntime) -> str:
    """
    返回天气情况
    Returns:
    - str: 天气情况描述
    """
    # 读取AgentState中的user_location
    city = runtime.state["user_location"]
    user_name = runtime.state["user_name"]
    return (
        f"你好{user_name},{city}的天气是晴天，温度20度，湿度60%"
        if city == "上海"
        else f"你好{user_name},{city}的天气是阴天，温度15度，湿度40%"
    )

@wrap_tool_call
def handle_tool_errors(request, handler):
    """当工具当前不可用时调用，返回自定义错误消息"""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Invalide User Param ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )

SYSTEM_PROMPT = dedent(
    f"""
你是一个用户生活助理，需要记住用户的信息，包括用户的姓名和位置
如果用户询问你天气，你需要根据用户的位置获取天气情况。
当在请求天气时没有用户的位置或名称时，请抛出错误
"""
).strip()

checkpointer = InMemorySaver()
config: RunnableConfig = {"configurable": {"thread_id": "1"}}


llm_agent = create_agent(
    model=chat_llm,
    tools=[
        greet,
        get_weather,
    ],
    middleware=[
        summarization_middleware,
        handle_tool_errors
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    state_schema=CustomAgentState,
)

# 设置初始值，但使用用户信息覆盖
llm_agent.invoke({"messages": [{"role":"user","content":"你好我是Cook，目前在上海"}],"user_name":"Alex","user_location":"北京"},config=config)
llm_agent.invoke({"messages": "请帮我介绍上海有哪些景点"}, config=config)
llm_agent.invoke({"messages": "请帮我介绍东方明珠"}, config=config)
final_response = llm_agent.invoke({"messages": "请帮我查询天气"}, config=config)
final_response["messages"][-1].pretty_print()
