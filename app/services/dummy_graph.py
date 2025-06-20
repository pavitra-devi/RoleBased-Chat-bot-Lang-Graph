from langgraph.graph import Graph,START, END
from langgraph.graph import StateGraph, add_messages
from langchain_core.messages import AnyMessage
from typing import TypedDict, Any, Annotated
from langchain.chat_models import init_chat_model
import os
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY", "AIzaSyCZt0Vrqv69IfvgzPId4WB_KpnIUIf8fJk")

# Model and embeddings
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
class AgentState(TypedDict):
    messages : Annotated[list, add_messages]

def call_grap(state:AgentState):
    response= model.invoke(state['messages'])
    state['messages'].append(response)
    return state
# Define the grap
graph=StateGraph(AgentState)
graph.add_node("engine", call_grap)
graph.add_edge(START, "engine")
graph.add_edge("engine", END)
agent= graph.compile()
result=agent.invoke({"messages": [{"role": "user", "content": "Hello, how are you?"}]})
print(result['messages'][-1].content)  # Should print the model's response