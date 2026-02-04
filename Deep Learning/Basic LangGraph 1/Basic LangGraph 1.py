#Message Type
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

h_msg = HumanMessage(content, name)
ai_msg = AIMessage(content, name)
sys_msg = SystemMessage(content, name)
tool_msg = ToolMessage(content, name,  tool_call_id, response_metadata)



#StateGraph
from langgraph.graph import START, END
from langgraph.graph import StateGraph, MessagesState

def bot_node(state: MessagesState) -> MessagesState:
    ai_resp = llm.invoke(state["messages"])

    return {"messages": [ai_resp]}

builder = StateGraph(MessagesState)
builder.add_node("bot", bot_node)
builder.add_edge(START, "bot")
builder.add_edge("bot", END)
graph = builder.compile()

initial_state = {"messages": [HumanMessage(content, name)]}
result = graph.invoke(initial_state)



#Tool
from langchain_core.tools import tool 

@tool
def tool_name(param: type):
    return result

tools = [tool_name]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

initial_state = {"messages": [HumanMessage(content, name)]}
result = graph.invoke(initial_state)



#Tool Condition
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def tool_name(param: type):
    return result

tools = [tool_name]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", tool_node)
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)
graph = builder.compile()

initial_state = {"messages": [HumanMessage(content, name)]}
result = graph.invoke(initial_state)