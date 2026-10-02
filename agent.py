import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi  # Ensures secure HTTPS connections by using trusted certificates

load_dotenv()

# Tell Python's HTTPS libraries to trust certifi's up-to-date CA certificates, which fixes "SSL: CERTIFICATE_VERIFY_FAILED" errors.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI   # Talks to Google's Gemini models
from langchain_core.messages import SystemMessage           # Hidden instruction message for the AI
from langgraph.graph import StateGraph, START, MessagesState  # Tools to build the agent's flowchart
from langgraph.prebuilt import ToolNode, tools_condition    # Ready-made "run tool" step and "needs a tool?" check
from langgraph.checkpoint.sqlite import SqliteSaver         # Saves chat state in a SQLite file
from tools import tools                                     # Your own list of tools (search, calculator, memory, RAG)


Path("data").mkdir(exist_ok=True)

DEFAULT_MODEL = os.getenv("GOOGLE_MODEL","gemini-2.5-flash")

# Only models in this safe list can be selected.
ALLOWED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",    # Included the lite version if needed
    "gemini-1.5-flash",         # Kept for fallback compatibility        
    "gemini-1.5-pro",
]


# The "rule book" sent to the AI before every conversation.
SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named PickGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using DuckDuckGo Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use DuckDuckGo Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""


def normalize_model_name(model_name : str | None) -> str :
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    # Safety check: ignore unknown model names.
    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def build_agent(model_name : str):
    """
    Build one LangGraph agent for a selected Gemini model.
    """

    selected_model = normalize_model_name(model_name)

    #Initialize ChatGoogleGenerativeAI
    # streaming: send words as they are generated.
    llm = ChatGoogleGenerativeAI(
        model = selected_model,
        temperature = 0.3,
        streaming = True
    )

    # Let Gemini know which tools it is allowed to ask for.
    llm_with_tools = llm.bind_tools(tools)

    # The "thinking" step: send rule book + chat history to Gemini and store its reply.
    def chatbot_node(state : MessagesState) :
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages" : [response]
        }

    # The "action" step: runs whichever tool Gemini asked for.
    tool_node = ToolNode(tools)

    # Start building the flowchart (graph) that shares chat messages between steps.
    workflow = StateGraph(MessagesState)

    workflow.add_node("chat_bot", chatbot_node)
    workflow.add_node("tools", tool_node)

    # Flow: START -> chat_bot
    workflow.add_edge(START,"chat_bot")
    # After chat_bot: go to "tools" if Gemini asked for a tool, otherwise finish.
    workflow.add_conditional_edges("chat_bot", tools_condition)
    # After a tool runs, go back to chat_bot so Gemini can use the result.
    workflow.add_edge("tools","chat_bot")

    # Database file that stores chat progress, so the bot remembers after restart.
    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    # Turn the flowchart into a working agent that saves its state with the checkpointer.
    return workflow.compile(checkpointer=checkpointer)


# Stores already-built agents so each model is built only once.
_AGENT_CACHE = {}

def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name=model_name)

    # Build the agent only the first time this model is requested.
    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]