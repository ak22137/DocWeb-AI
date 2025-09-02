import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from PyPDF2 import PdfReader

# Load API keys
def load_api_keys():
    """Load API keys from APIs.txt file"""
    
    with open("APIs.txt", "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == "googleGemini":
                    os.environ["GOOGLE_API_KEY"] = parts[1]
                elif parts[0] == "tavily":
                    os.environ["TAVILY_API_KEY"] = parts[1]
        
   

load_api_keys()

# Initialize model
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Tools
@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    search = TavilySearch(max_results=3)
    return search.invoke(query)

@tool
def read_resume() -> str:
    """Read the resume PDF file"""
    try:
        reader = PdfReader("Docs/akshatRESUME3.2.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return f"📄 Resume:\n{text}"
    except Exception as e:
        return f"❌ Error: {e}"

@tool
def read_file(file_path: str) -> str:
    """Read any file"""
    try:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            with open(file_path, "r") as f:
                return f.read()
    except Exception as e:
        return f"❌ Error: {e}"

@tool
def ask_human(question: str) -> str:
    """Ask human for input"""
    return interrupt({"query": question})["data"]

# Agent
def agent_node(state: AgentState):
    system_msg = SystemMessage(content="""
    You are a helpful AI assistant. You can:
    - Search the web for information
    - Read and analyze the resume PDF
    - Read other files
    - Ask humans for help
    
    For shopping questions, search for product info and give recommendations.
    For resume questions, read the resume first.
    """)
    
    messages = [system_msg] + state["messages"]
    tools = [search_web, read_resume, read_file, ask_human]
    response = model.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([search_web, read_resume, read_file, ask_human]))
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# Compile
graph = graph.compile(checkpointer=MemorySaver())

# Run
if __name__ == "__main__":
    print("🤖 Simple AI Assistant")
    config = {"thread_id": "chat"}
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"):
                if "messages" in event:
                    print(f"\n🤖 {event['messages'][-1].content}")
        
        except KeyboardInterrupt:
            break
    
    print("👋 Bye!")
