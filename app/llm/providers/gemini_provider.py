"""
Gemini LLM Provider Implementation

Uses Google's gemini-3.5-flash model.
Supports Tool Calling by translating generic schemas to Gemini's protobuf format.
"""
import asyncio
import base64
import json
import uuid
import google.generativeai as genai
from typing import AsyncGenerator, Any
import structlog

from app.llm.base import LLMProvider, LLMRequest, LLMResponse, LLMMessage, ToolDefinition
from app.config.settings import get_settings

logger = structlog.get_logger()


def _protobuf_to_dict(obj: Any) -> Any:
    """
    Recursively convert Google's MapComposite/RepeatedComposite objects 
    into standard Python dicts/lists so they can be JSON serialized.
    """
    if hasattr(obj, "items") and callable(obj.items):
        # It's a MapComposite (dict-like)
        return {k: _protobuf_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        # It's a RepeatedComposite (list-like)
        return [_protobuf_to_dict(v) for v in obj]
    else:
        # It's a primitive (string, int, float, bool)
        return obj


class GeminiProvider(LLMProvider):
    """Google Gemini LLM Provider implementation."""
    
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model_id = settings.gemini_model
        self._model = genai.GenerativeModel(self.model_id)
        self.logger = logger.bind(provider="gemini", model=self.model_id)
    
    def _convert_tools(self, tools: list[ToolDefinition]) -> list:
        """Convert generic OpenAI-style tools to Gemini's protobuf format."""
        if not tools:
            return []
            
        function_declarations = []
        for tool in tools:
            func = tool.function
            params = func.get("parameters", {})
            
            def convert_type(t):
                mapping = {
                    "string": "STRING",
                    "number": "NUMBER",
                    "integer": "INTEGER",
                    "boolean": "BOOLEAN",
                    "array": "ARRAY",
                    "object": "OBJECT"
                }
                return mapping.get(t.lower(), "STRING") if t else "STRING"
                
            schema = {
                "type": convert_type(params.get("type", "object")),
                "properties": {},
                "required": params.get("required", [])
            }
            
            for prop_name, prop_val in params.get("properties", {}).items():
                prop_schema = {
                    "type": convert_type(prop_val.get("type", "string")),
                    "description": prop_val.get("description", "")
                }
                
                if prop_schema["type"] == "ARRAY" and "items" in prop_val:
                    prop_schema["items"] = {
                        "type": convert_type(prop_val["items"].get("type", "string"))
                    }
                    
                schema["properties"][prop_name] = prop_schema
                
            function_declarations.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": schema
            })
            
        return [{"function_declarations": function_declarations}]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate complete response from Gemini."""
        log = self.logger.bind(request_id=request.request_id)
        
        system_prompt, contents = self._format_messages(request.messages)
        gemini_tools = self._convert_tools(request.tools) if request.tools else None
        
        model = self._model
        if system_prompt or gemini_tools:
            model = genai.GenerativeModel(
                self.model_id, 
                system_instruction=system_prompt,
                tools=gemini_tools
            )
        
        generation_config = genai.GenerationConfig(
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        log.debug("Calling Gemini API")
        response = await asyncio.to_thread(
            model.generate_content,
            contents,
            generation_config=generation_config,
        )
        
        content = ""
        finish_reason = "unknown"
        tool_calls = None
        
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = str(candidate.finish_reason)
            
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content += part.text
                    elif hasattr(part, 'function_call') and part.function_call.name:
                        if not tool_calls:
                            tool_calls = []
                        
                        func_name = part.function_call.name
                        
                        # Use our custom converter to handle MapComposite safely
                        args_dict = _protobuf_to_dict(part.function_call.args)
                        args_json = json.dumps(args_dict)
                        
                        # Capture thought_signature from the Part (not function_call) for Gemini 3 series
                        # The thought_signature is a bytes object attached to the Part
                        thought_signature = None
                        if hasattr(part, 'thought_signature') and part.thought_signature:
                            # thought_signature is bytes, encode to base64 string for transport
                            thought_signature = base64.b64encode(part.thought_signature).decode('utf-8')
                            log.debug("Captured thought_signature for function call", 
                                     func_name=func_name, has_signature=bool(thought_signature))
                        
                        tool_call_entry = {
                            "id": part.function_call.id if hasattr(part.function_call, 'id') and part.function_call.id else f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": args_json
                            }
                        }
                        
                        # Include thought_signature if present
                        if thought_signature:
                            tool_call_entry["thought_signature"] = thought_signature
                        
                        tool_calls.append(tool_call_entry)
        
        usage = response.usage_metadata
        
        log.info(
            "Gemini response received",
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            finish_reason=finish_reason,
            has_tool_calls=bool(tool_calls)
        )
        
        return LLMResponse(
            content=content if content else None,
            model=self.model_id,
            provider="gemini",
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            finish_reason=finish_reason,
            tool_calls=tool_calls
        )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream tokens from Gemini."""
        system_prompt, contents = self._format_messages(request.messages)
        gemini_tools = self._convert_tools(request.tools) if request.tools else None
        
        model = self._model
        if system_prompt or gemini_tools:
            model = genai.GenerativeModel(
                self.model_id, 
                system_instruction=system_prompt,
                tools=gemini_tools
            )
        
        generation_config = genai.GenerationConfig(
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        response = await asyncio.to_thread(
            model.generate_content,
            contents,
            generation_config=generation_config,
            stream=True,
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def _format_messages(self, messages: list[LLMMessage]) -> tuple[str | None, list]:
        """
        Convert standard messages to Gemini format.
        Translates OpenAI-style multi-message tool flows into Gemini's format.
        """
        system_prompt = None
        contents = []
        func_name_map = {}
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                
            elif msg.role == "assistant" and msg.tool_calls:
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})
                    
                for tc in msg.tool_calls:
                    func_name = tc["function"]["name"]
                    tool_call_id = tc["id"]
                    func_name_map[tool_call_id] = func_name
                    
                    args_dict = json.loads(tc["function"]["arguments"])
                    function_call_part = {
                        "function_call": {
                            "name": func_name,
                            "args": args_dict
                        }
                    }
                    
                    # Include thought_signature at Part level (not inside function_call)
                    # The thought_signature is a base64-encoded string that must be decoded to bytes
                    if "thought_signature" in tc and tc["thought_signature"]:
                        # Decode base64 string back to bytes for the API
                        function_call_part["thought_signature"] = base64.b64decode(tc["thought_signature"])
                    
                    parts.append(function_call_part)
                contents.append({"role": "model", "parts": parts})
                
            elif msg.role == "tool":
                func_name = func_name_map.get(msg.tool_call_id, "unknown_function")
                contents.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": func_name,
                            "response": {"result": msg.content}
                        }
                    }]
                })
                
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})
        
        return system_prompt, contents
    
    def get_model_id(self) -> str:
        return self.model_id
    
    async def health_check(self) -> bool:
        try:
            test_request = LLMRequest(
                messages=[LLMMessage(role="user", content="Reply with: OK")],
                max_tokens=10,
            )
            response = await self.generate(test_request)
            return len(response.content or "") > 0
        except Exception:
            return False