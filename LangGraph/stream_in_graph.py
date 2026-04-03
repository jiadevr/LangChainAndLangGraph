# [LangGraph Streaming Tokens](https://docs.langchain.com/oss/python/langgraph/streaming#llm-tokens)
from langgraph import graph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, content
from langchain_openai import ChatOpenAI
import os
from typing import TypedDict


class CustomState(TypedDict):
    topic: str
    content: str


chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.5-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)


def call_llm(state: CustomState):
    reponse = chat_llm.invoke([HumanMessage(content=f"请帮我简单介绍{state['topic']}")])
    return {"content": reponse.content}


graph_builder = StateGraph(CustomState)
graph_builder.add_node("llm_call", call_llm)
graph_builder.add_edge(START, "llm_call")
graph_builder.add_edge("llm_call", END)

stream_graph = graph_builder.compile()
# stream_result=stream_graph.invoke({"topic":"trae和cursor的差别"})
# print(stream_result)

print("______________stream_result______________")
# 这种形式已经非常类似在线LLM的输出
for chunk in stream_graph.stream(
    {"topic": "trae和cursor的差别"}, stream_mode="messages", version="v2"
):
    # print(chunk)
    # print("______________chunk______________")
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)
# 返回值示意
# {
#     "type": "messages",
#     "ns": (),
#     "data": (
#         AIMessageChunk(
#             content="",
#             additional_kwargs={},
#             response_metadata={},
#             id="lc_run--019d5277-4571-74d0-ab9a-fe8ff745b4cf",
#             tool_calls=[],
#             invalid_tool_calls=[],
#             tool_call_chunks=[],
#             chunk_position="last",
#         ),
#         {
#             "ls_integration": "langchain_chat_model",
#             "langgraph_step": 1,
#             "langgraph_node": "llm_call",
#             "langgraph_triggers": ("branch:to:llm_call",),
#             "langgraph_path": ("__pregel_pull", "llm_call"),
#             "langgraph_checkpoint_ns": "llm_call:156956d7-1f59-b954-59a8-a7acd322bf09",
#             "checkpoint_ns": "llm_call:156956d7-1f59-b954-59a8-a7acd322bf09",
#             "ls_provider": "openai",
#             "ls_model_name": "qwen3.5-plus",
#             "ls_model_type": "chat",
#             "ls_temperature": 0.7,
#             "ls_max_tokens": 1000,
#         },
#     ),
# }
