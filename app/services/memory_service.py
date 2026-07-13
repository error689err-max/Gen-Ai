"""
Memory Service

Manages conversation history in Redis.
Stores messages as JSON strings and enforces a TTL so old sessions expire.
"""
import json
import structlog
from app.db.redis_client import get_redis_client
from app.llm.base import LLMMessage
from app.config.settings import get_settings

logger = structlog.get_logger()


class MemoryService:
    """Handles loading and saving conversation history."""
    
    def __init__(self):
        self.redis = get_redis_client()
        self.settings = get_settings()
    
    def _get_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"
    
    async def get_history(self, session_id: str) -> list[LLMMessage]:
        """Load conversation history for a session."""
        key = self._get_key(session_id)
        data = await self.redis.lrange(key, 0, -1)
        
        messages = []
        for item in data:
            msg_dict = json.loads(item)
            messages.append(LLMMessage(**msg_dict))
            
        return messages
    
    async def save_turn(self, session_id: str, user_msg: str, assistant_msg: str, tool_calls: list = None) -> None:
        """Save a user query and assistant response to history.
        
        Args:
            session_id: The session identifier
            user_msg: The user's message
            assistant_msg: The assistant's textual response
            tool_calls: Optional list of tool calls (needed for thought_signature preservation)
        """
        key = self._get_key(session_id)
        
        user_msg_obj = LLMMessage(role="user", content=user_msg).model_dump()
        
        # Create assistant message - include tool_calls if present (for thought_signature preservation)
        assistant_msg_obj = LLMMessage(
            role="assistant", 
            content=assistant_msg,
            tool_calls=tool_calls
        ).model_dump()
        
        # Push to Redis list
        await self.redis.rpush(key, json.dumps(user_msg_obj))
        await self.redis.rpush(key, json.dumps(assistant_msg_obj))
        
        # Trim list to last 10 messages (5 turns) to save memory
        await self.redis.ltrim(key, -10, -1)
        
        # Set TTL so history expires if unused
        await self.redis.expire(key, self.settings.session_ttl_seconds)

    async def save_conversation(self, session_id: str, messages: list) -> None:
        """Save the full conversation (including tool calls) to history.
        
        This replaces the existing history with the full conversation.
        Needed for preserving thought_signatures in Gemini 3 series.
        Tool calls must be preserved across turns for the API to work correctly.
        
        Args:
            session_id: The session identifier
            messages: List of LLMMessage objects representing the full conversation
        """
        key = self._get_key(session_id)
        
        # Delete existing history first (replace instead of append)
        await self.redis.delete(key)
        
        # Convert all messages to JSON and save
        for msg in messages:
            if isinstance(msg, LLMMessage):
                msg_obj = msg.model_dump()
            else:
                msg_obj = msg
            await self.redis.rpush(key, json.dumps(msg_obj))
        
        # Trim list to last 20 messages (10 turns) to save memory
        # We keep more messages here because tool calls add overhead
        await self.redis.ltrim(key, -20, -1)
        
        # Set TTL so history expires if unused
        await self.redis.expire(key, self.settings.session_ttl_seconds)