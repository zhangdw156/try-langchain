import os
from typing import List, TypedDict, Union, Annotated

from langchain.agents import AgentExecutor, create_openai_tools_agent
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# ==============================================================================
# 1. 环境设置
# ==============================================================================
load_dotenv()
# 从 .env 文件加载配置
api_key = os.environ.get("LOCAL_API_KEY")
base_url = os.environ.get("OPENAI_API_BASE")
model_name = os.environ.get("MODEL_NAME")


# ==============================================================================
# 2. 定义状态管理 (State) - 保持不变
# ==============================================================================
class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AIMessage, str]
    intermediate_steps: List[Annotated[Union[AIMessage, ToolMessage], "Intermediate steps"]]
    reflections: List[str]
    verifications: List[str]


# ==============================================================================
# 3. 定义工具 (Tools) - 核心改动
# ==============================================================================
# 使用 Tavily Search，它更稳定可靠
# max_results=3 表示每次搜索最多返回3条结果
print("正在初始化搜索工具: TavilySearchResults")
tools = [TavilySearch(max_results=3)]

# ==============================================================================
# 4. 定义 Agent 角色 - 保持不变
# ==============================================================================

# --- 4.1. LLM 实例 ---
actor_llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
reflector_llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
verifier_llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)

# --- 4.2. Actor (执行者) ---
actor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个强大的人工智能助手，擅长使用工具来寻找问题的答案。
请尽力完成任务，并仔细考虑之前的反思和验证反馈。
如果你已经找到了最终答案，请直接以最终答案的形式回复，不要再调用任何工具。""",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="history"),
    ]
)
actor_agent = create_openai_tools_agent(actor_llm, tools, actor_prompt)
actor_executor = AgentExecutor(agent=actor_agent, tools=tools, verbose=True)

# --- 4.3. Reflector (反思者) ---
reflector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位 Agent 行为分析专家。你的任务是分析一个 Agent 失败的执行轨迹，并提供具体的、有建设性的改进建议。
请只输出你需要 Agent 在下一次尝试中记住的关键反思，不要有任何其他多余的文字。""",
        ),
        ("user", "这是 Agent 上次失败的完整思考过程：\n{intermediate_steps}"),
    ]
)
reflector_chain = reflector_prompt | reflector_llm

# --- 4.4. Verifier (验证者) ---
verifier_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位事实核查员和答案评估专家。你的任务是评估一个 AI Agent 的最终答案。
请将 Agent 的答案与用户的原始问题进行比较，并检查以下几点：
1. 答案是否完整地回答了问题的所有部分？
2. 答案是否准确无误？
3. 答案是否清晰、简洁？

如果答案没有问题，请只回复 "OK"。
如果答案有问题，请提供具体的、可操作的修改建议，告诉 Agent 应该如何修正它的答案。""",
        ),
        ("user", "原始问题: {input}\n\nAgent 的答案: {agent_outcome}"),
    ]
)
verifier_chain = verifier_prompt | verifier_llm


# ==============================================================================
# 5. 主控制循环 - (修正版)
# ==============================================================================

def run_agent_with_reflection_and_verification(
        input_question: str, max_loops: int = 3
):
    state = AgentState(
        input=input_question,
        agent_outcome="",
        intermediate_steps=[],
        reflections=[],
        verifications=[],
    )

    for loop in range(max_loops):
        print(f"\n{'=' * 20} 循环: {loop + 1} {'=' * 20}")

        print("\n--- [ 阶段 1: Actor 执行 ] ---")
        history: List[BaseMessage] = []
        if state["reflections"]:
            history.append(
                HumanMessage(
                    content=f"这是之前失败后的反思，请采纳：\n" + "\n".join(state["reflections"])
                )
            )
        if state["verifications"]:
            history.append(
                HumanMessage(
                    content=f"这是对上次答案的验证反馈，请采纳：\n" + "\n".join(state["verifications"])
                )
            )

        result = actor_executor.invoke({
            "input": state["input"],
            "history": history
        })

        # ==================== 调试代码开始 ====================
        print("\n" + "*" * 10 + " DEBUGGING RESULT " + "*" * 10)
        print(f"Type of result: {type(result)}")
        print(f"Keys in result: {list(result.keys())}")

        intermediate_steps_value = result.get("intermediate_steps")
        print(f"Type of intermediate_steps from result: {type(intermediate_steps_value)}")
        print(f"Content of intermediate_steps from result: {intermediate_steps_value}")
        print("*" * 30 + "\n")
        # ==================== 调试代码结束 ====================

        # --- 关键修正 ---
        # 使用 .get() 来安全地访问 'intermediate_steps'
        # 如果它不存在（因为 Agent 没有调用工具），就返回一个空列表 []
        state["agent_outcome"] = result.get("output", "")
        state["intermediate_steps"] = result.get("intermediate_steps", [])

        print("\n--- [ 阶段 2: Verifier 验证 ] ---")
        # ... 后续代码保持不变 ...
        verification_result = verifier_chain.invoke({
            "input": state["input"],
            "agent_outcome": state["agent_outcome"],
        }).content.strip()

        print(f"验证结果: {verification_result}")
        if "OK" in verification_result.upper():
            print("\n--- [ 任务成功，验证通过！ ] ---")
            return state["agent_outcome"]
        else:
            state["verifications"].append(verification_result)

        print("\n--- [ 阶段 3: Reflector 反思 ] ---")
        # 如果没有 intermediate_steps，Reflector 可能无法很好地工作
        # 但至少程序不会崩溃。我们可以让它基于最终答案进行反思。
        # 一个更好的做法是，如果 intermediate_steps 为空，就将 agent_outcome 也传给 reflector
        if not state["intermediate_steps"]:
            print("警告: 没有工具调用轨迹可供反思，将基于最终输出来进行反思。")
            reflection_input = f"Agent 没有调用任何工具，直接输出了以下内容：\n{state['agent_outcome']}"
        else:
            reflection_input = str(state["intermediate_steps"])

        reflection_result = reflector_chain.invoke({
            "intermediate_steps": reflection_input,
        }).content.strip()

        print(f"反思建议: {reflection_result}")
        state["reflections"].append(reflection_result)

    print(f"\n--- [ 达到最大循环次数 ({max_loops})，任务失败 ] ---")
    return f"无法在限定次数内完成任务。最终答案是：\n{state['agent_outcome']}"


# ==============================================================================
# 6. 运行示例
# ==============================================================================
if __name__ == "__main__":
    complex_question = "分析一下 COVID-19 疫情期间，美联储的量化宽松政策（QE）是如何通过影响半导体供应链，最终传导到普通消费者的游戏机（如 PS5）购买价格上的？请梳理出完整的因果链条。"

    final_answer = run_agent_with_reflection_and_verification(complex_question)

    print("\n\n################# 最终答案 #################")
    print(final_answer)
