from langgraph.graph import StateGraph, START, END
from utils.logger import logger
from pydantic import BaseModel, Field
from typing import List, Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[List, add_messages]


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


##################################################
builder = StateGraph(State)
logger.info(f"builder: {builder}")
logger.info(f"builder.schemas: {builder.schemas}")
##################################################
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
##################################################
llm = ChatOpenAI(model="Qwen3-14B",
                 api_key="key",
                 base_url="http://127.0.0.1:49004/v1")
# logger.info(f"模型调用: {llm.invoke('你是谁')}")
##################################################
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
logger.info(f"graph: {graph}")
##################################################
thread_config = {
    "configurable": {
        "thread_id": "123"
    }
}
response = graph.invoke({"messages": [{"role": "human", "content": "你好，我是张三"}]}, thread_config)
logger.info(response["messages"])
logger.info(response["messages"][-1])
response = graph.invoke({"messages": [{"role": "human", "content": "我是谁"}]}, thread_config)
logger.info(response["messages"])
logger.info(response["messages"][-1])
