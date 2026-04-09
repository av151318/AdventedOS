import json

from proxyapp.parsers.factory_content_tools import extract_factory_tool_calls_from_raw
from proxyapp.parsers.nemotron_content_tools import extract_promotable_tool_calls_from_raw


def test_factory_todo_write():
    raw = """\n<function=TodoWrite>\n<parameter=todos>\n1. [in_progress] a\n2. [pending] b\n</parameter>\n</function=TodoWrite>"""
    tools, pfx = extract_factory_tool_calls_from_raw(raw)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "TodoWrite"
    args = json.loads(tools[0]["function"]["arguments"])
    assert "todos" in args
    assert "[in_progress]" in args["todos"]


def test_promotable_prefers_nemotron_when_present():
    raw = (
        "<tool_call>\n<function=ping>\n<parameter=x>1</parameter>\n</function>\n</tool_call>\n"
        "<function=Other>\n<parameter=a>b</parameter>\n</function=Other>"
    )
    tools, _ = extract_promotable_tool_calls_from_raw(raw)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "ping"


def test_promotable_factory_when_no_tool_call():
    raw = "Hi <function=Execute><parameter=command>date</parameter></function=Execute> tail"
    tools, pfx = extract_promotable_tool_calls_from_raw(raw)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "Execute"
    assert pfx.strip() == "Hi"
