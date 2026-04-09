import json
import time
import uuid


class ChatCompletionEventBuilder:
    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self.created = int(time.time())

    def chunk(
        self,
        content_delta: str | None = None,
        reasoning_delta: str | None = None,
        tool_calls_delta: list[dict] | None = None,
        finish_reason: str | None = None,
        index: int = 0,
    ) -> dict:
        delta: dict = {}
        if reasoning_delta:
            delta["reasoning_content"] = reasoning_delta
        if content_delta:
            delta["content"] = content_delta
        if tool_calls_delta:
            delta["tool_calls"] = tool_calls_delta

        choice = {
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason,
        }

        return {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }

    def final_chunk(
        self,
        content: str = "",
        reasoning: str = "",
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
        index: int = 0,
        usage: dict | None = None,
    ) -> dict:
        message: dict = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls

        choice = {
            "index": index,
            "message": message,
            "finish_reason": finish_reason,
        }

        result = {
            "id": self.response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }
        if usage:
            result["usage"] = usage
        return result
