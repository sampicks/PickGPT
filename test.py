from agent import get_agent
from langchain_core.messages import AIMessageChunk, HumanMessage
from database import init_db

from tools import set_current_thread_id

init_db()

agent = get_agent("gemini-2.5-flash")

config = {
    "configurable" : {
        "thread_id" : "test_thread_id"
    }
}


thread_id = config["configurable"]["thread_id"]

set_current_thread_id(thread_id)


for message_chunk, metadata in agent.stream(
        {'messages' : [HumanMessage(content="what is my name")]},
        # {'messages' : [HumanMessage(content="my name is Peeyoosh Soamya Khare")]},
        config = config,
        stream_mode = 'messages'):
    
    # Only stream assistant tokens from the chatbot node, skipping ToolMessages
    if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
        if isinstance(message_chunk.content, str):
            print(message_chunk.content, end="", flush=True)
    print()  # newline after stream completes