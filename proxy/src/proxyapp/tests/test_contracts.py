import pytest
from proxyapp.contracts.responses import ResponsesEventBuilder
from proxyapp.contracts.chat_completions import ChatCompletionEventBuilder
from proxyapp.contracts.backend_adapter import build_backend_chat_request_from_responses
from proxyapp.streaming.state import ResponseState
from proxyapp.sanitizers.reasoning_xml import ReasoningXmlSanitizer
from proxyapp.proxy import maybe_promote_nemotron_tools_in_completion, normalize_vllm_chat_completion_response


class TestResponseState:
    def test_unique_ids(self):
        s1 = ResponseState(model="test")
        s2 = ResponseState(model="test")
        assert s1.response_id != s2.response_id

    def test_sequence_increments(self):
        state = ResponseState(model="test")
        assert state.next_sequence() == 1
        assert state.next_sequence() == 2


class TestResponsesEventBuilder:
    def test_created(self):
        state = ResponseState(model="m")
        b = ResponsesEventBuilder(response_id=state.response_id, model="m")
        e = b.created()
        assert e["type"] == "response.created"
        assert e["response"]["id"] == state.response_id

    def test_in_progress(self):
        state = ResponseState(model="m")
        b = ResponsesEventBuilder(response_id=state.response_id, model="m")
        e = b.in_progress()
        assert e["type"] == "response.in_progress"

    def test_output_item_added(self):
        state = ResponseState(model="m")
        b = ResponsesEventBuilder(response_id=state.response_id, model="m")
        e = b.output_item_added(0, {"id": "x", "type": "message", "role": "assistant", "content": []})
        assert e["type"] == "response.output_item.added"
        assert e["output_index"] == 0

    def test_output_text_delta(self):
        state = ResponseState(model="m")
        b = ResponsesEventBuilder(response_id=state.response_id, model="m")
        e = b.output_text_delta("x", 0, "Hello")
        assert e["type"] == "response.output_text.delta"
        assert e["delta"] == "Hello"

    def test_completed(self):
        state = ResponseState(model="m")
        b = ResponsesEventBuilder(response_id=state.response_id, model="m")
        e = b.completed([{"type": "message"}])
        assert e["type"] == "response.completed"
        assert e["response"]["status"] == "completed"

    def test_failed(self):
        state = ResponseState(model="m")
        b = ResponsesEventBuilder(response_id=state.response_id, model="m")
        e = b.failed("err")
        assert e["type"] == "response.failed"
        assert e["response"]["error"]["message"] == "err"

    def test_sequence_monotonic(self):
        state = ResponseState(model="m")
        b = ResponsesEventBuilder(response_id=state.response_id, model="m")
        seqs = [b.created()["sequence_number"], b.in_progress()["sequence_number"], b.completed([])["sequence_number"]]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == 3


class TestChatCompletionEventBuilder:
    def test_content_delta(self):
        b = ChatCompletionEventBuilder(response_id="c1", model="m")
        chunk = b.chunk(content_delta="Hi")
        assert chunk["choices"][0]["delta"]["content"] == "Hi"

    def test_reasoning_delta(self):
        b = ChatCompletionEventBuilder(response_id="c1", model="m")
        chunk = b.chunk(reasoning_delta="Think")
        assert chunk["choices"][0]["delta"]["reasoning_content"] == "Think"

    def test_tool_calls_delta(self):
        b = ChatCompletionEventBuilder(response_id="c1", model="m")
        tc = [{"id": "t1", "type": "function", "function": {"name": "s", "arguments": ""}}]
        chunk = b.chunk(tool_calls_delta=tc)
        assert chunk["choices"][0]["delta"]["tool_calls"] == tc

    def test_final_chunk(self):
        b = ChatCompletionEventBuilder(response_id="c1", model="m")
        final = b.final_chunk(content="Done", finish_reason="stop")
        assert final["choices"][0]["message"]["content"] == "Done"
        assert final["choices"][0]["finish_reason"] == "stop"


class TestBackendAdapter:
    def test_string_input(self):
        r = build_backend_chat_request_from_responses({"input": "Hi", "model": "t"}, "/app/model")
        assert r["model"] == "/app/model"
        assert r["messages"][0] == {"role": "user", "content": "Hi"}

    def test_list_input(self):
        r = build_backend_chat_request_from_responses({"input": [{"role": "user", "content": "Hi"}], "model": "t"}, "/app/model")
        assert r["messages"][0] == {"role": "user", "content": "Hi"}

    def test_tools(self):
        tools = [{"type": "function", "function": {"name": "s", "parameters": {}}}]
        r = build_backend_chat_request_from_responses({"input": "Hi", "tools": tools}, "/app/model")
        assert r["tools"] == tools

    def test_tool_choice(self):
        r = build_backend_chat_request_from_responses({"input": "Hi", "tool_choice": "auto"}, "/app/model")
        assert r["tool_choice"] == "auto"

    def test_instructions(self):
        r = build_backend_chat_request_from_responses({"input": "Hi", "instructions": "Be brief"}, "/app/model")
        assert r["messages"][0] == {"role": "system", "content": "Be brief"}
        assert r["messages"][1] == {"role": "user", "content": "Hi"}


class TestReasoningXmlSanitizer:
    def test_clean_content(self):
        s = ReasoningXmlSanitizer()
        _, c = s.feed("content", "Hello world")
        assert c == "Hello world"

    def test_xml_stripped(self):
        s = ReasoningXmlSanitizer()
        tag_open = chr(60) + "invoke name=x" + chr(62)
        tag_close = chr(60) + "/invoke" + chr(62)
        _, c = s.feed("content", "A " + tag_open + "1" + tag_close + " B")
        assert "invoke" not in c
        assert "A " in c
        assert " B" in c

    def test_reasoning_passthrough(self):
        s = ReasoningXmlSanitizer()
        r, _ = s.feed("reasoning", "Thinking...")
        assert r == "Thinking..."

    def test_partial_tag_held(self):
        s = ReasoningXmlSanitizer()
        _, c = s.feed("content", "Hello " + chr(60))
        assert c == "Hello "
        _, c2 = s.feed("content", "invoke name=x" + chr(62) + "1" + chr(60) + "/invoke" + chr(62) + " world")
        assert "invoke" not in (c2 or "")
        assert " world" in (c2 or "")

    def test_drain_releases_content(self):
        s = ReasoningXmlSanitizer()
        _, c = s.feed("content", "Hello ")
        assert c == "Hello "
        assert s.drain() is None

    def test_drain_discards_incomplete_xml(self):
        s = ReasoningXmlSanitizer()
        s.feed("content", "Hello " + chr(60) + "invoke name=x")
        assert s.drain() is None

    def test_multiple_xml_blocks(self):
        s = ReasoningXmlSanitizer()
        tag_open = chr(60) + "invoke name=x" + chr(62)
        tag_close = chr(60) + "/invoke" + chr(62)
        _, c = s.feed("content", "A " + tag_open + "1" + tag_close + " B " + tag_open + "2" + tag_close + " C")
        assert "invoke" not in c
        assert c == "A  B  C"

    def test_plain_angle_bracket(self):
        s = ReasoningXmlSanitizer()
        _, c = s.feed("content", "x < 5 and y > 3")
        assert c == "x < 5 and y > 3"

    def test_nemotron_tool_call_block_stripped(self):
        s = ReasoningXmlSanitizer()
        o = chr(60) + "tool_call" + chr(62)
        c = chr(60) + "/tool_call" + chr(62)
        inner = (
            chr(60)
            + "function=demo"
            + chr(62)
            + chr(60)
            + "parameter=p"
            + chr(62)
            + "v"
            + chr(60)
            + "/parameter"
            + chr(62)
            + chr(60)
            + "/function"
            + chr(62)
        )
        _, v = s.feed("content", "Hi " + o + inner + c + " tail")
        assert "tool_call" not in (v or "").lower()
        assert "parameter" not in (v or "").lower()
        assert "Hi " in (v or "")
        assert "tail" in (v or "")

    def test_tool_call_split_chunks(self):
        s = ReasoningXmlSanitizer()
        o = chr(60) + "tool_call" + chr(62)
        c = chr(60) + "/tool_call" + chr(62)
        _, v1 = s.feed("content", "A" + o + "bod")
        _, v2 = s.feed("content", "y" + c + "B")
        assert "tool_call" not in ((v1 or "") + (v2 or "")).lower()
        assert "A" in ((v1 or "") + (v2 or "")) and "B" in ((v1 or "") + (v2 or ""))

    def test_tool_call_whitespace_before_gt_close(self):
        s = ReasoningXmlSanitizer()
        o = chr(60) + "tool_call" + chr(62)
        # Model often emits space before closing angle bracket
        c = chr(60) + "/tool_call " + chr(62)
        inner = "inner"
        _, v = s.feed("content", "A" + o + inner + c + " B")
        merged = (v or "").lower()
        assert "tool_call" not in merged
        assert "/tool" not in merged
        assert "A" in (v or "") and " B" in (v or "")

    def test_tool_call_split_whitespace_close(self):
        s = ReasoningXmlSanitizer()
        o = chr(60) + "tool_call" + chr(62)
        _, v1 = s.feed("content", "A" + o + "body" + chr(60) + "/tool_call ")
        assert "tool_call" not in (v1 or "").lower()
        _, v2 = s.feed("content", chr(62) + "Z")
        out = ((v1 or "") + (v2 or "")).lower()
        assert "tool_call" not in out
        assert "/tool" not in out
        assert "z" in out

    def test_factory_function_block_stripped(self):
        s = ReasoningXmlSanitizer()
        raw = (
            "X <function=TodoWrite><parameter=todos>a</parameter></function=TodoWrite> Y"
        )
        _, v = s.feed("content", raw)
        assert "function" not in (v or "").lower()
        assert "parameter" not in (v or "").lower()
        assert "X " in (v or "") and " Y" in (v or "")

    def test_think_split_across_chunks(self):
        s = ReasoningXmlSanitizer()
        o = chr(60) + "think" + chr(62)
        c = chr(60) + "/think" + chr(62)
        r1, v1 = s.feed("content", o + "partial")
        r2, v2 = s.feed("content", " middle" + c + "visible")
        merged_r = (r1 or "") + (r2 or "")
        merged_v = (v1 or "") + (v2 or "")
        assert "partial" in merged_r and "middle" in merged_r
        assert merged_v == "visible"

    def test_redacted_case_insensitive(self):
        s = ReasoningXmlSanitizer()
        tag_o = chr(60) + "REDACTED_THINKING" + chr(62)
        tag_c = chr(60) + "/REDACTED_THINKING" + chr(62)
        r, v = s.feed("content", tag_o + "a" + tag_c + "b")
        assert (r or "").strip() == "a"
        assert v == "b"

    def test_finalize_unclosed_reasoning(self):
        s = ReasoningXmlSanitizer()
        r, _ = s.feed("content", chr(60) + "think" + chr(62) + "no close")
        assert "no close" in (r or "")
        rr, vv = s.finalize()
        assert rr is None
        assert vv is None


class TestVllmChatCompletionNormalize:
    def test_copies_reasoning_to_content_when_content_null(self):
        body = {"choices": [{"message": {"role": "assistant", "content": None, "reasoning": "HELLO"}}]}
        out = normalize_vllm_chat_completion_response(body)
        assert out["choices"][0]["message"]["content"] == "HELLO"

    def test_reasoning_content_field(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "", "reasoning_content": "RC"}}]}
        out = normalize_vllm_chat_completion_response(body)
        assert out["choices"][0]["message"]["content"] == "RC"

    def test_noop_when_content_present(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "xY", "reasoning": "z"}}]}
        out = normalize_vllm_chat_completion_response(body)
        assert out["choices"][0]["message"]["content"] == "xY"


class TestNemotronPromotionNonStream:
    def test_promotes_tool_xml_to_tool_calls(self, monkeypatch):
        monkeypatch.setenv("PROXY_STRICT_OAI_TOOLS", "1")
        monkeypatch.delenv("PROXY_PROMOTE_CONTENT_TOOLS_WHEN_REQUESTED", raising=False)
        req = {"tools": [{"type": "function", "function": {"name": "do_ping", "parameters": {}}}]}
        raw_content = (
            "ok "
            "<tool_call>\n<function=do_ping>\n<parameter=x>1</parameter>\n</function>\n</tool_call>\n"
        )
        body = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": raw_content}}]}
        out = maybe_promote_nemotron_tools_in_completion(body, req)
        msg = out["choices"][0]["message"]
        assert msg.get("tool_calls") and msg["tool_calls"][0]["function"]["name"] == "do_ping"
        assert msg.get("content") == "ok"
        assert out["choices"][0]["finish_reason"] == "tool_calls"

    def test_skips_without_tools_declared(self, monkeypatch):
        monkeypatch.setenv("PROXY_STRICT_OAI_TOOLS", "1")
        body = {"choices": [{"message": {"role": "assistant", "content": "<tool_call></tool_call>"}}]}
        out = maybe_promote_nemotron_tools_in_completion(body, {})
        assert out["choices"][0]["message"].get("tool_calls") is None

    def test_promotes_factory_function_markup(self, monkeypatch):
        monkeypatch.setenv("PROXY_STRICT_OAI_TOOLS", "1")
        req = {"tools": [{"type": "function", "function": {"name": "TodoWrite", "parameters": {}}}]}
        content = "<function=TodoWrite>\n<parameter=todos>1. ok\n</parameter>\n</function=TodoWrite>"
        body = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}
        out = maybe_promote_nemotron_tools_in_completion(body, req)
        assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "TodoWrite"
        assert out["choices"][0]["finish_reason"] == "tool_calls"
