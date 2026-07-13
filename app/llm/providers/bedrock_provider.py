# app/llm/providers/bedrock_provider.py
"""
Amazon Bedrock LLM Provider Implementation

Uses AWS Bedrock to access various LLM models (Claude, Titan, Llama, etc.)
"""
import json
import boto3
from typing import AsyncGenerator
import structlog

from app.llm.base import LLMProvider, LLMRequest, LLMResponse, LLMMessage
from app.config.settings import get_settings

logger = structlog.get_logger()


class BedrockProvider(LLMProvider):
    """Amazon Bedrock LLM Provider implementation."""
    
    def __init__(self):
        settings = get_settings()
        self.region_name = settings.aws_region
        self.model_id = settings.bedrock_model_id
        
        # Initialize boto3 bedrock runtime client
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name
        )
        
        self.logger = logger.bind(provider="bedrock", model=self.model_id)
        
        # Map model IDs to their API format
        self.model_map = {
            # Anthropic Claude models
            "anthropic.claude-3-opus-20240229-v1:0": self._claude_format,
            "anthropic.claude-3-sonnet-20240229-v1:0": self._claude_format,
            "anthropic.claude-3-5-sonnet-20241022-v2:0": self._claude_format,
            "anthropic.claude-3-haiku-20240307-v1:0": self._claude_format,
            "anthropic.claude-2.1": self._claude_format,
            "anthropic.claude-2": self._claude_format,
            # Other models use the default format
        }
    
    def _claude_format(self, request: LLMRequest) -> dict:
        """Format request for Claude models on Bedrock."""
        system_prompt = None
        messages = []
        
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                messages.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})
            elif msg.role == "assistant":
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"])
                        })
                messages.append({"role": "assistant", "content": content})
            elif msg.role == "tool":
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": [{"type": "text", "text": msg.content}]
                    }]
                })
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages
        }
        
        if system_prompt:
            body["system"] = [{"type": "text", "text": system_prompt}]
        
        # Add tool use for Claude 3 models
        if request.tools:
            tools = []
            for t in request.tools:
                func = t.function
                tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
            body["tools"] = tools
        
        return body
    
    def _default_format(self, request: LLMRequest) -> dict:
        """Default format for non-Claude models (Titan, Llama, etc.)."""
        system_prompt = None
        messages = []
        
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content or ""})
        
        body = {
            "inputText": messages[-1]["content"] if messages else "",
            "textGenerationConfig": {
                "maxTokenCount": request.max_tokens,
                "temperature": request.temperature,
            }
        }
        
        return body
    
    def _parse_claude_response(self, response_body: str) -> LLMResponse:
        """Parse Claude response from Bedrock."""
        response = json.loads(response_body)
        
        content = None
        tool_calls = None
        finish_reason = "unknown"
        
        if "content" in response:
            for block in response["content"]:
                if block.get("type") == "text":
                    content = block.get("text", "")
                elif block.get("type") == "tool_use":
                    if not tool_calls:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    })
        
        if "stop_reason" in response:
            finish_reason = response["stop_reason"]
        
        usage = response.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=self.model_id,
            provider="bedrock",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            finish_reason=finish_reason,
            tool_calls=tool_calls
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from Bedrock."""
        log = self.logger.bind(request_id=getattr(request, 'request_id', None))
        
        log.debug("Calling Bedrock API", model=self.model_id)
        
        # Get the appropriate formatter for this model
        formatter = self.model_map.get(self.model_id, self._default_format)
        body = formatter(request)
        
        # Call Bedrock
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = response.get("body").read().decode("utf-8")
        
        # Parse response
        llm_response = self._parse_claude_response(response_body)
        
        log.info(
            "Bedrock response received",
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            finish_reason=llm_response.finish_reason,
            has_tool_calls=bool(llm_response.tool_calls)
        )
        
        return llm_response
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream tokens from Bedrock."""
        log = self.logger.bind(request_id=getattr(request, 'request_id', None))
        
        formatter = self.model_map.get(self.model_id, self._default_format)
        body = formatter(request)
        
        # Invoke with streaming
        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        
        for event in response.get("body"):
            if event.get("chunk"):
                chunk = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                if "content_block_delta" in chunk:
                    delta = chunk["content_block_delta"]
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")
    
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
        except Exception as e:
            self.logger.error("Bedrock health check failed", error=str(e))
            return False
