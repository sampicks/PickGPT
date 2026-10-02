import os
from dotenv import load_dotenv
import certifi  # Ensures secure HTTPS connections by using trusted certificates

load_dotenv()

# Tell Python's HTTPS libraries to trust certifi's up-to-date CA certificates, which fixes "SSL: CERTIFICATE_VERIFY_FAILED" errors.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path
import re
import ast
import json

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage, AIMessage, AIMessageChunk, ToolMessage
)

from agent import get_agent
from database import (
    init_db, save_chat_message, get_chat_history, create_or_update_conversation, list_conversations
)
from rag import add_document_to_rag
from tools import set_current_thread_id


app = FastAPI()

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


init_db()


@app.get("/")
async def home(request : Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/conversations")
async def conversations():
    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }


@app.get("/history/{thread_id}")
async def history(thread_id: str):
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id, "Uploaded document")

        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id
        )

        return JSONResponse({
            "success": True,
            "message": f"Uploaded {result['filename']} and created {result['chunks']} chunks."
        })

    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )


@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400
        )

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", "gemini-2.5-flash")

    if not user_message.strip():
        return JSONResponse(
            {"error": "Message is required."},
            status_code=400
        )

    agent = get_agent(selected_model)

    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", user_message)

    set_current_thread_id(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    def event_generator():
        final_answer = ""

        try:
            inputs = {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            }

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages"
            ):
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)

            yield sse_data({"done": True})

        except Exception as e:
            # Parse the exception into a clean user-facing string
            clean_error = format_stream_exception(e)

            # Option A: Send clean error as a token so it renders directly in the chat message UI
            yield sse_data({"token": clean_error})

            # Option B: Or send as error key if your UI handles {"error": ...} payload specifically
            # yield sse_data({"error": clean_error})

            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    """
    Prevents tool/search/RAG raw data from streaming, 
    but ALLOWS error messages to pass through to the frontend.
    """
    # 1. ALWAYS allow error strings/objects so the frontend receives them
    if is_api_error(chunk):
        return True

    metadata = metadata or {}
    node_name = str(metadata.get("langgraph_node", "")).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}

    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    """
    Formats clean text chunks or maps raw API errors into human-readable text.
    """
    # 1. If chunk is an error string/exception object, format it
    if is_api_error(chunk):
        return format_api_error(chunk)

    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        # Check if text content itself is an error string
        if is_api_error(content):
            return format_api_error(content)
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        combined_text = "".join(text_parts)
        if is_api_error(combined_text):
            return format_api_error(combined_text)
        return combined_text

    return ""


def is_api_error(chunk) -> bool:
    """Detects if the chunk or string contains an API error message."""
    chunk_str = str(chunk)
    error_indicators = ["RESOURCE_EXHAUSTED", "429", "'error':", '"error":', "Quota exceeded"]
    return any(indicator in chunk_str for indicator in error_indicators)

def format_api_error(error_input) -> str:
    """
    Parses raw error strings or dicts and converts them into user-friendly messages.
    """
    error_str = str(error_input)
    
    # 1. Try to extract retry time if available (e.g., '33s' or '33.002899486s')
    retry_match = re.search(r"retry in (\d+(?:\.\d+)?s|\d+s)", error_str, re.IGNORECASE)
    retry_info = f" Please try again in {retry_match.group(1)}." if retry_match else ""

    # 2. Check for 429 Quota Exceeded
    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
        return f"⚠️ API rate limit or quota reached.{retry_info}"
    
    # 3. Handle Other Common Errors
    if "500" in error_str or "INTERNAL" in error_str:
        return "⚠️ The AI service is currently experiencing high load or internal issues. Please try again shortly."
    if "401" in error_str or "UNAUTHENTICATED" in error_str:
        return "⚠️ Authentication error. Please check your API key settings."

    # Generic Fallback
    return "⚠️ An unexpected API error occurred. Please try again."

def format_stream_exception(e: Exception) -> str:
    """Formats raw API exceptions into clean, user-friendly error messages."""
    err_str = str(e)

    # Handle 429 Quota Exceeded / Rate Limit
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        # Extract retry delay if available (e.g., '33s' or '33.002899486s')
        match = re.search(r"retry in (\d+(?:\.\d+)?s|\d+s)", err_str, re.IGNORECASE)
        retry_suffix = f" Please try again in {match.group(1)}." if match else ""
        return f"⚠️ API quota or rate limit reached.{retry_suffix}"

    # Handle authentication issues
    if "401" in err_str or "UNAUTHENTICATED" in err_str:
        return "⚠️ Authentication failed. Please check your API key settings."

    # Handle server issues
    if "500" in err_str or "INTERNAL" in err_str:
        return "⚠️ The AI service is currently experiencing high load. Please try again shortly."

    # Fallback for other errors
    return "⚠️️ An unexpected error occurred while processing your request."


if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
