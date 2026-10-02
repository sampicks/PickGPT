import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from database import save_memory, search_memory
from rag import retrieve_from_rag



load_dotenv()

CURRENT_THREAD_ID = "default"

def set_current_thread_id(thread_id : str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


# Here we don't need to add @tool decorator because DuckDuckGoSearchResults is already a tool
web_search = DuckDuckGoSearchResults(
    max_results = 5,
    topic = "general",          # "general", "news", "images"
    search_depth = "advanced"   # "basic", "advanced"
)

# A safe calculator tool: evaluates math expressions using only allowed functions
@tool
def calculator(expression: str) -> str :
    """
    Useful for simple math calculations. 
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(9), 10 * 5
    """

    try:
        allowed = {
            "math" : math,
            "abs" : abs,
            "round" : round,
            "min" : min,
            "max" : max,
            "sum" : sum
        }

        result = eval(expression,{"__builtins__":{}}, allowed)
        return str(result)
    
    except Exception as e:
        return f"Calculation error : {str(e)}"


@tool
def remember_this(memory: str) -> str :
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return save_memory(thread_id=CURRENT_THREAD_ID,memory=memory)


@tool
def recall_memory(query: str) -> str :
    """
    Recall saved long-term memories about the user of this converation.
    """

    return search_memory(thread_id=CURRENT_THREAD_ID,query= query)


@tool
def search_uploaded_documents(query : str) -> str :
    """
    Search uploaded doucments for relevant information.
    Use this when the user ask about uploaded PDFs, DOCX, TXT, files or documents.
    """

    return retrieve_from_rag(query=query, thread_id=CURRENT_THREAD_ID)

tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]