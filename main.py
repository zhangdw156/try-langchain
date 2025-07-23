from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from utils.logger import logger
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


# 1. (推荐) 定义你期望的 JSON 结构
class Joke(BaseModel):
    joke_content: str = Field(description="笑话的内容")
    explanation: str = Field(description="笑话的解释，说明笑点在哪里")


# 2. 创建一个 JsonOutputParser，并传入你的 Pydantic 模型
output_parser = JsonOutputParser(pydantic_object=Joke)
# 3. 创建一个包含格式化指令的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个善于思考、善于反思的AI助手，熟悉阿里、腾讯、字节、百度和美团的职场知识。"),
    ("human", "{query}"),
    MessagesPlaceholder(variable_name="history")

])
# 4. 实例化 LLM
llm = ChatOpenAI(model="Qwen3-14B",
                 api_key="key",
                 base_url="http://127.0.0.1:49004/v1")
tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)
# 5. 构建链
chain = prompt | llm_with_tools

# 6. 调用链
query = "3 * 12等于多少？ 11 + 49呢？"
messages = [HumanMessage(query)]
result = chain.invoke({"query": query, "history": []})
logger.info(result)

messages.append(result)

for tool_call in result.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

logger.info(messages)

result = chain.invoke({"query": query, "history": messages})
logger.info(result)
