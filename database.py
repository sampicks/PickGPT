from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Create a "data" folder if it doesn't exist (exist_ok=True = no error if it already exists).
Path("data").mkdir(exist_ok=True)

# SQLite keeps the whole database in a single file: data/chatbot_memory.db
DATABASE_URL = "sqlite:///data/chatbot_memory.db"

# The engine is the connection to the database file.
# check_same_thread=False lets web apps (Streamlit/FastAPI) use SQLite from multiple threads.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# SessionLocal is a "session factory": every SessionLocal() call opens a fresh
# working session for reading/writing the database.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Base is the parent class of all our table classes below.
Base = declarative_base()


# Each class below = one table in the database. Each Column = one column in that table.
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    # unique=True: no two chats can share the same thread_id.
    # index=True: makes searching by this column faster.
    thread_id = Column(String, unique=True, index=True)
    title = Column(String, default="New Chat")
    # No () after utcnow on purpose: it runs for every new row, not once at startup.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    # thread_id links each message to its chat (not unique: one chat has many messages).
    thread_id = Column(String, index=True)
    role = Column(String)  # for example "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    memory = Column(Text)  # a fact worth remembering, e.g. "User likes Python"
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    # Creates the tables inside the database file if they don't exist yet.
    # Call this once when your app starts.
    Base.metadata.create_all(bind=engine)


def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    db = SessionLocal()

    try:
        # Look for an existing chat with this thread_id (.first() gives None if not found).
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if not conversation:
            title = "New Chat"

            if first_message:
                # Use the first 40 characters of the first message as the chat title.
                title = first_message.strip()[:40]
                if len(first_message.strip()) > 40:
                    title += "..."

            conversation = Conversation(
                thread_id=thread_id,
                title=title,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # add() only stages the new row; nothing is saved until commit().
            db.add(conversation)

        else:
            conversation.updated_at = datetime.utcnow()

        # commit() permanently writes the changes to the database file.
        db.commit()

    finally:
        db.close()


def list_conversations():
    db = SessionLocal()

    try:
        return (
            db.query(Conversation)
            # desc() = newest first, so recent chats appear at the top of the sidebar.
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    finally:
        db.close()


def save_chat_message(thread_id: str, role: str, content: str):
    db = SessionLocal()

    try:
        msg = ChatMessage(
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=datetime.utcnow()
        )

        db.add(msg)

        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        # Bump the chat's "last updated" time so it moves to the top of the list.
        if conversation:
            conversation.updated_at = datetime.utcnow()

        db.commit()

    finally:
        db.close()


def get_chat_history(thread_id: str):
    db = SessionLocal()

    try:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            # asc() = oldest first, so the chat reads in the correct order.
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

    finally:
        db.close()


def save_memory(thread_id: str, memory: str):
    db = SessionLocal()

    try:
        item = LongTermMemory(
            thread_id=thread_id,
            memory=memory,
            created_at=datetime.utcnow()
        )

        db.add(item)
        db.commit()

        return "Memory saved successfully."

    finally:
        db.close()


def search_memory(thread_id: str, query: str):
    db = SessionLocal()

    try:
        # Note: "query" is not used yet. This simply returns the 20 most recent memories.
        memories = (
            db.query(LongTermMemory)
            .filter(LongTermMemory.thread_id == thread_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(20)
            .all()
        )

        if not memories:
            return "No saved memory found."

        # Join all memories into one text block, one "- memory" bullet per line,
        # which is easy to pass to the AI model.
        return "\n".join([f"- {m.memory}" for m in memories])

    finally:
        db.close()