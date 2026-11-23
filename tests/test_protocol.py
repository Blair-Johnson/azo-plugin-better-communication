"""End-to-end Sans-I/O protocol tests for better_communication."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utils.core.events import InputReceived
from agent_utils.core.session import Session
from agent_zoo.pipelines.default import build_default_builder

import better_communication as plugin


def _build_session(tmp_path: Path, monkeypatch, *, min_trigger_words: int = 5, rewrite_stages=None) -> Session:
    cfg = json.loads((Path(plugin.__file__).parents[1] / "better_communication.json").read_text())
    cfg["enabled"] = True
    cfg["triggers"]["min_trigger_words"] = min_trigger_words
    cfg["triggers"]["assistant_messages"] = 2
    if rewrite_stages is not None:
        cfg["rewrite_stages"] = rewrite_stages
    path = tmp_path / "better_communication.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv("AZO_BETTER_COMMUNICATION_CONFIG", str(path))

    session = Session(system_prompt="Base prompt.")
    builder = build_default_builder(
        session,
        config={},
        include_plugins=False,
        include_llm_drivers=False,
    )
    plugin.register_features(builder, session=session, config={})
    session.pipeline = builder.build()
    session.ensure_initialized()
    return session


def _tool_call_msg(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def test_system_prompt_guidance_rendered_on_model_channel(tmp_path, monkeypatch):
    session = _build_session(tmp_path, monkeypatch)
    session.step(InputReceived(message={"role": "user", "content": "Please investigate."}))
    messages = session.render()
    assert "# Message Composition" in messages[0]["content"]
    assert "record_obligation" in messages[0]["content"]


def test_obligation_record_list_close_lifecycle(tmp_path, monkeypatch):
    session = _build_session(tmp_path, monkeypatch)
    session.step(InputReceived(message={"role": "user", "content": "Check tests and report."}))

    session.step(
        InputReceived(
            message=_tool_call_msg(
                "c1",
                "record_obligation",
                {
                    "user_language": "Check tests",
                    "predicted_response_expectations": "Report pass/fail counts",
                },
            )
        )
    )
    obl_id = session.state.entries[-1].tool_calls[0].result
    assert obl_id.startswith("obl_")

    session.step(InputReceived(message=_tool_call_msg("c2", "list_obligations", {})))
    listed = session.state.entries[-1].tool_calls[0].result
    assert obl_id in listed and "Check tests" in listed

    session.step(
        InputReceived(
            message=_tool_call_msg(
                "c3",
                "close_obligation",
                {"obligation_id": obl_id, "reason": "Handled directly"},
            )
        )
    )
    assert "Removed " + obl_id in session.state.entries[-1].tool_calls[0].result
    assert len(getattr(session.state, plugin._OBLIGATIONS)) == 0


def test_interception_rewrite_and_publish_protocol(tmp_path, monkeypatch):
    stages = [
        {"name": "coverage", "prompt": "Check coverage."},
        {"name": "polish", "prompt": "Polish prose."},
    ]
    session = _build_session(tmp_path, monkeypatch, min_trigger_words=3, rewrite_stages=stages)
    session.step(InputReceived(message={"role": "user", "content": "Refactor and verify."}))
    session.render()

    # Record one obligation.
    session.step(
        InputReceived(
            message=_tool_call_msg(
                "c_obl",
                "record_obligation",
                {
                    "user_language": "Refactor and verify",
                    "predicted_response_expectations": "Outcome and test status",
                },
            )
        )
    )
    obl_id = session.state.entries[-1].tool_calls[0].result

    # Plain-text assistant reply should be intercepted because assistant_messages >= 2 and words >= 3.
    session.step(
        InputReceived(
            message={
                "role": "assistant",
                "content": "Initial draft explaining refactor results clearly.",
            }
        )
    )
    intercepted_entry = session.state.entries[-1]
    assert intercepted_entry.tool_calls
    assert intercepted_entry.tool_calls[0].name == "compose_message"
    assert not intercepted_entry.tool_calls[0].error
    assert "Simulated human experience across time" in intercepted_entry.tool_calls[0].result
    assert "Rewrite stage 1 of 2: coverage" in intercepted_entry.tool_calls[0].result

    buffer_id = getattr(session.state, plugin._ACTIVE_BUFFER)
    assert buffer_id

    # Advancing without editing fails with exact ERROR prefix.
    session.step(
        InputReceived(
            message=_tool_call_msg(
                "c_rw_fail",
                "rewrite_response",
                {"buffer": buffer_id, "closing_obligations": [obl_id]},
            )
        )
    )
    assert session.state.entries[-1].tool_calls[0].error is True
    assert "unchanged" in session.state.entries[-1].tool_calls[0].result

    # Stage 1 edit + rewrite_response
    session.state.buffer_manager.get(buffer_id).text = "Stage 1 revised draft with full coverage."
    session.step(
        InputReceived(
            message=_tool_call_msg(
                "c_rw1",
                "rewrite_response",
                {"buffer": buffer_id, "closing_obligations": [obl_id]},
            )
        )
    )
    rw1 = session.state.entries[-1].tool_calls[0]
    assert rw1.error is False
    assert "Rewrite stage 2 of 2: polish" in rw1.result

    # Stage 2 edit + rewrite_response
    session.state.buffer_manager.get(buffer_id).text = "Stage 2 final polished message for the user."
    session.step(
        InputReceived(
            message=_tool_call_msg(
                "c_rw2",
                "rewrite_response",
                {"buffer": buffer_id},
            )
        )
    )
    rw2 = session.state.entries[-1].tool_calls[0]
    assert rw2.error is False
    assert "Submission is unlocked" in rw2.result

    # Publish via message_user
    session.step(
        InputReceived(
            message=_tool_call_msg(
                "c_pub",
                "message_user",
                {"buffer": buffer_id},
            )
        )
    )
    published_entry = session.state.entries[-1]
    assert published_entry.role == "assistant"
    assert not published_entry.tool_calls
    assert published_entry.messages[0]["content"] == "Stage 2 final polished message for the user."
    assert published_entry.metadata["better_communication"]["published"] is True
    assert published_entry.metadata["better_communication"]["closed_obligations"] == [obl_id]
    assert len(getattr(session.state, plugin._OBLIGATIONS)) == 0
    assert getattr(session.state, plugin._ACTIVE_BUFFER) == ""
    assert session.needs_reply is False
