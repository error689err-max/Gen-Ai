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

# Maximum content length to prevent payload size issues
MAX_CONTENT_LENGTH = 4000
MAX_HISTORY_MESSAGES = 6  # 3 turns (user + assistant + tool calls)


class MemoryService:
    """Handles loading and saving conversation history."""
    
    def __init__(self):
        self.redis = get_redis_client()
        self.settings = get_settings()
    
    def _get_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"
    
    async def get_history(self, session_id: str) -> list[LLMMessage]:
        """Load conversation history for a session.
        
        Returns limited history (last 2 turns) to prevent payload size issues.
        """
        key = self._get_key(session_id)
        data = await self.redis.lrange(key, 0, -1)
        
        messages = []
        for item in data:
            msg_dict = json.loads(item)
            messages.append(LLMMessage(**msg_dict))
        
        # Limit to last 2 turns (4 messages) to prevent payload size issues
        return messages[-4:]
    
    async def save_turn(self, session_id: str, user_msg: str, assistant_msg: str, tool_calls: list = None) -> None:
        """Save a user query and assistant response to history.
        
        Args:
            session_id: The session identifier
            user_msg: The user's message
            assistant_msg: The assistant's textual response
            tool_calls: Optional, not used for Groq (kept for API compatibility)
        """
        key = self._get_key(session_id)
        
        # Truncate content if too long
        user_content = user_msg[:MAX_CONTENT_LENGTH] if user_msg else ""
        assistant_content = assistant_msg[:MAX_CONTENT_LENGTH] if assistant_msg else ""
        
        user_msg_obj = LLMMessage(role="user", content=user_content).model_dump()
        assistant_msg_obj = LLMMessage(role="assistant", content=assistant_content).model_dump()
        
        # Push to Redis list
        await self.redis.rpush(key, json.dumps(user_msg_obj))
        await self.redis.rpush(key, json.dumps(assistant_msg_obj))
        
        # Trim list to last MAX_HISTORY_MESSAGES (alternating user/assistant)
        await self.redis.ltrim(key, -MAX_HISTORY_MESSAGES, -1)
        
        # Set TTL so history expires if unused
        await self.redis.expire(key, self.settings.session_ttl_seconds)

    async def save_conversation(self, session_id: str, messages: list) -> None:
        """Save the conversation to history with size limits.
        
        Truncates long content and limits message count to prevent
        payload size issues with external APIs.
        
        Args:
            session_id: The session identifier
            messages: List of LLMMessage objects representing the conversation
        """
        key = self._get_key(session_id)
        
        # Delete existing history first
        await self.redis.delete(key)
        
        # Keep only recent messages (alternating user/assistant pairs)
        # Also truncate long content
        messages_to_save = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                msg_obj = msg.model_dump()
            else:
                msg_obj = msg
            
            # Truncate long content
            if msg_obj.get("content") and len(msg_obj["content"]) > MAX_CONTENT_LENGTH:
                msg_obj["content"] = msg_obj["content"][:MAX_CONTENT_LENGTH] + "... [truncated]"
            
            # Truncate tool call arguments if too long
            if msg_obj.get("tool_calls"):
                for tc in msg_obj["tool_calls"]:
                    if tc.get("function", {}).get("arguments"):
                        args = tc["function"]["arguments"]
                        if len(args) > 2000:
                            tc["function"]["arguments"] = args[:2000] + '"...[truncated]'
            
            messages_to_save.append(msg_obj)
        
        # Keep only the last MAX_HISTORY_MESSAGES
        messages_to_save = messages_to_save[-MAX_HISTORY_MESSAGES:]
        
        # Save messages
        for msg_obj in messages_to_save:
            await self.redis.rpush(key, json.dumps(msg_obj))
        
        logger.info("Saved conversation", 
                   session_id=session_id, 
                   message_count=len(messages_to_save))
        
        # Set TTL
        await self.redis.expire(key, self.settings.session_ttl_seconds)