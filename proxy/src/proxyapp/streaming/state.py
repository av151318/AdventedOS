import uuid
import time


class ResponseState:
    def __init__(self, model: str):
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.model = model
        self.sequence_number = 0
        self.status = "in_progress"
        self.created_at = int(time.time())
        self.output_index = 0

    def next_sequence(self) -> int:
        self.sequence_number += 1
        return self.sequence_number


class MessageItemState:
    def __init__(self, output_index: int):
        self.item_id = f"msg_{uuid.uuid4().hex[:16]}"
        self.output_index = output_index
        self.announced = False
        self.content_index = 0
        self.text = ""

    def next_content_index(self) -> int:
        idx = self.content_index
        self.content_index += 1
        return idx


class ReasoningItemState:
    def __init__(self, output_index: int):
        self.item_id = f"reason_{uuid.uuid4().hex[:16]}"
        self.output_index = output_index
        self.announced = False
        self.content_index = 0
        self.text = ""

    def next_content_index(self) -> int:
        idx = self.content_index
        self.content_index += 1
        return idx


class FunctionCallItemState:
    def __init__(self, output_index: int, call_id: str, name: str):
        self.item_id = f"fc_{uuid.uuid4().hex[:16]}"
        self.call_id = call_id
        self.name = name
        self.arguments = ""
        self.output_index = output_index
        self.announced = False
        self.done = False
