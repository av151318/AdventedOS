"""Admission / crash classifier / retry eligibility helpers."""

from proxyapp.streaming.committed_ledger import StreamCommittedLedger, is_retry_eligible
from proxyapp.streaming.crash_classify import (
    classify_stream_termination,
    is_client_disconnect_message,
    should_trip_admission_on_classify,
    stream_idle_timeout_seconds,
)


def test_stream_idle_default():
    assert stream_idle_timeout_seconds() >= 45.0


def test_client_disconnect_message():
    assert is_client_disconnect_message(
        "Cannot write to closing transport"
    )
    assert not is_client_disconnect_message("Connection reset by peer")


def test_classify_idle_stall():
    c = classify_stream_termination(
        exc=None, saw_done=False, mid_stream=True, idle_timeout=True
    )
    assert c == "upstream_stall"
    assert should_trip_admission_on_classify(c)


def test_classify_client_disconnect():
    c = classify_stream_termination(
        exc=RuntimeError("cannot write to closing transport"),
        saw_done=False,
        mid_stream=True,
    )
    assert c == "client_disconnect"
    assert not should_trip_admission_on_classify(c)


def test_is_retry_eligible_read_only():
    ledger = StreamCommittedLedger(request_id="r1")
    assert is_retry_eligible(ledger, {"tools": None})
    assert is_retry_eligible(ledger, {"tools": []})
    ledger.last_tool_event = {"index": 0, "partial": True}
    assert not is_retry_eligible(ledger, {"tools": []})
    ledger.last_tool_event = None
    assert not is_retry_eligible(ledger, {"tools": [{"type": "function"}]})
