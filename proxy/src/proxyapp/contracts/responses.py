import uuid
import json


class ResponsesEventBuilder:
    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _base(self, event_type: str) -> dict:
        return {"type": event_type, "sequence_number": self._next_seq()}

    def created(self) -> dict:
        return {
            **self._base("response.created"),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self._seq,
                "status": "in_progress",
                "model": self.model,
                "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        }

    def in_progress(self) -> dict:
        return {
            **self._base("response.in_progress"),
            "response": {"id": self.response_id, "status": "in_progress"},
        }

    def output_item_added(self, output_index: int, item: dict) -> dict:
        return {
            **self._base("response.output_item.added"),
            "output_index": output_index,
            "item": item,
        }

    def content_part_added(self, item_id: str, content_index: int, part_type: str = "output_text") -> dict:
        return {
            **self._base("response.content_part.added"),
            "item_id": item_id,
            "content_index": content_index,
            "part": {"type": part_type},
        }

    def output_text_delta(self, item_id: str, content_index: int, delta: str) -> dict:
        return {
            **self._base("response.output_text.delta"),
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta,
        }

    def output_text_done(self, item_id: str, content_index: int, text: str) -> dict:
        return {
            **self._base("response.output_text.done"),
            "item_id": item_id,
            "content_index": content_index,
            "text": text,
        }

    def content_part_done(self, item_id: str, content_index: int) -> dict:
        return {
            **self._base("response.content_part.done"),
            "item_id": item_id,
            "content_index": content_index,
            "part": {"type": "output_text"},
        }

    def output_item_done(self, item_id: str) -> dict:
        return {
            **self._base("response.output_item.done"),
            "item_id": item_id,
        }

    def reasoning_text_delta(self, item_id: str, delta: str) -> dict:
        return {
            **self._base("response.reasoning_text.delta"),
            "item_id": item_id,
            "delta": delta,
        }

    def reasoning_text_done(self, item_id: str, text: str) -> dict:
        return {
            **self._base("response.reasoning_text.done"),
            "item_id": item_id,
            "text": text,
        }

    def function_call_arguments_delta(self, call_id: str, delta: str) -> dict:
        return {
            **self._base("response.function_call_arguments.delta"),
            "call_id": call_id,
            "delta": delta,
        }

    def function_call_arguments_done(self, call_id: str, name: str, arguments: str) -> dict:
        return {
            **self._base("response.function_call_arguments.done"),
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }

    def completed(self, output: list) -> dict:
        return {
            **self._base("response.completed"),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "completed",
                "model": self.model,
                "output": output,
            },
        }

    def failed(self, error: str) -> dict:
        return {
            **self._base("response.failed"),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "failed",
                "model": self.model,
                "output": [],
                "error": {"message": error},
            },
        }
