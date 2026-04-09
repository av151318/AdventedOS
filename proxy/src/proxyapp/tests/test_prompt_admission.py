"""Tests for rough prompt token estimation."""

from __future__ import annotations

from proxyapp.prompt_admission import (
    AdmissionThresholds,
    admission_class_slot,
    classify_admission_class,
    estimate_chat_prompt_tokens,
    is_first_chat_turn,
)


def test_is_first_turn_no_assistant() -> None:
    assert is_first_chat_turn([{"role": "user", "content": "hi"}]) is True
    assert is_first_chat_turn([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]) is True


def test_is_first_turn_with_assistant() -> None:
    assert (
        is_first_chat_turn(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        is False
    )


def test_estimate_positive() -> None:
    msgs = [{"role": "user", "content": "a" * 400}]
    t = estimate_chat_prompt_tokens(msgs, None)
    assert t >= 100


def test_admission_class_buckets() -> None:
    th = AdmissionThresholds(short_lt=8192, medium_lt=32768, long_lt=98304)
    assert classify_admission_class(1000, th) == "short"
    assert classify_admission_class(10000, th) == "medium"
    assert classify_admission_class(50000, th) == "long"
    assert classify_admission_class(100000, th) == "extreme"
    assert admission_class_slot(1000, th) == "short"
    assert admission_class_slot(50000, th) == "heavy"
    assert admission_class_slot(100000, th) == "heavy"
