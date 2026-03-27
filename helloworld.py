from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
import os

chatLLM = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus",
    streaming=True,
)
res = chatLLM.stream([HumanMessage(content="hi")], streaming=True)
for r in res:
    print("chat resp:", r.content)