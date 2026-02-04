#Reducer
import operator

class Chatbot(TypedDict):
    messages: Annotated[list, add_messages]
    counter: Annotated[int, operator.add]

def assistant(state: Chatbot):
    enhanced_sys_msg = SystemMessage(content)
    messages = [enhanced_sys_msg]+state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response], "counter": 1}
    
    
    
#MessageSaver
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

messages = [HumanMessage(content)]

config = {"configurable": {"thread_id": "1"}}
result = react_graph_memory.invoke({"messages": messages}, config)



#Invoke & Stream
model = ChatOpenAI(model="gpt-4o-mini", streaming=True)

for chunk in model.stream(prompt):
    print(chunk.content, end="")