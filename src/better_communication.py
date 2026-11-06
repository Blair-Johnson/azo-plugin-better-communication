"""Guard human attention with an explicit draft-preview-submit protocol."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
import stat
import tempfile
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from agent_zoo.runtime.settings import Setting, SettingError

from agent_utils import Entry, Feature, Tool, ToolCall, estimate_tokens
from agent_utils.components import (
    AppendInput,
    CompressToolResults,
    ConsolidateToolResults,
    ExcludeForgotten,
    MessageRenderer,
    RenderTransformRegistrar,
    ToolDispatchStart,
    TurnCounter,
)
from agent_utils.types import MODEL_RENDER_CHANNEL
from agents.system_prompt import SystemPromptSkillList

COMPOSE_TOOL = "compose_message"
REWRITE_TOOL = "rewrite_response"
MESSAGE_USER_TOOL = "message_user"
RECORD_OBLIGATION_TOOL = "record_obligation"
LIST_OBLIGATIONS_TOOL = "list_obligations"
CLOSE_OBLIGATION_TOOL = "close_obligation"
_TURN_STARTED_AT = "_communication_turn_started_at"
_TURN_USER_ENTRY_INDEX = "_communication_turn_user_entry_index"
_ACTIVE_BUFFER = "_communication_active_buffer"
_OBLIGATIONS = "_communication_obligations"
_REWRITES = "_communication_rewrites"
_PENDING_PUBLICATIONS = "_communication_pending_publications"
_CONFIG_ENV = "AZO_BETTER_COMMUNICATION_CONFIG"
_CONFIG_NAME = "better_communication.json"
_ENTRY_METADATA_KEY = "better_communication"
_DEFERRED_EAGER_TOOL_NAMES = "_deferred_eager_tool_names"
_COMMUNICATION_TOOL_NAMES = frozenset({
    COMPOSE_TOOL,
    REWRITE_TOOL,
    MESSAGE_USER_TOOL,
    RECORD_OBLIGATION_TOOL,
    LIST_OBLIGATIONS_TOOL,
    CLOSE_OBLIGATION_TOOL,
})
class RewriteStage:
    __slots__ = ("name", "prompt")

    def __init__(self, name: str, prompt: str):
        self.name = name
        self.prompt = prompt


class Obligation:
    __slots__ = (
        "obligation_id",
        "user_language",
        "predicted_response_expectations",
        "source_user_message",
        "source_entry_index",
        "added_at_time",
        "added_at_message_count",
    )

    def __init__(
        self,
        obligation_id: str,
        user_language: str,
        predicted_response_expectations: str,
        source_user_message: str,
        source_entry_index: int,
        added_at_time: float,
        added_at_message_count: int,
    ):
        self.obligation_id = obligation_id
        self.user_language = user_language
        self.predicted_response_expectations = predicted_response_expectations
        self.source_user_message = source_user_message
        self.source_entry_index = source_entry_index
        self.added_at_time = added_at_time
        self.added_at_message_count = added_at_message_count

    def to_dict(self) -> dict[str, object]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Obligation":
        return cls(**data)


class ObligationList:
    """Ordered push/remove list of user-facing communication obligations."""

    def __init__(self):
        self._items: list[Obligation] = []

    def add(
        self,
        *,
        user_language: str,
        predicted_response_expectations: str,
        source_user_message: str,
        source_entry_index: int,
        message_count: int,
    ) -> str:
        obligation_id = f"obl_{uuid.uuid4().hex[:10]}"
        self._items.append(Obligation(
            obligation_id=obligation_id,
            user_language=user_language,
            predicted_response_expectations=predicted_response_expectations,
            source_user_message=source_user_message,
            source_entry_index=source_entry_index,
            added_at_time=time.time(),
            added_at_message_count=message_count,
        ))
        return obligation_id

    def ids(self) -> tuple[str, ...]:
        return tuple(item.obligation_id for item in self._items)

    def remove(self, obligation_id: str) -> Obligation | None:
        for index, item in enumerate(self._items):
            if item.obligation_id == obligation_id:
                return self._items.pop(index)
        return None

    def remove_many(self, obligation_ids: list[str]) -> list[Obligation]:
        removed = []
        for obligation_id in obligation_ids:
            item = self.remove(obligation_id)
            if item is not None:
                removed.append(item)
        return removed

    def render(self, *, current_time: float, current_message_count: int) -> str:
        if not self._items:
            return "No obligations are currently recorded."
        lines = ["Obligations:"]
        for item in self._items:
            elapsed = _format_duration(current_time - item.added_at_time)
            messages = max(0, current_message_count - item.added_at_message_count)
            lines.extend([
                f"- {item.obligation_id} (age: {elapsed}; {messages} messages)",
                f"  User language: {item.user_language}",
                f"  Predicted response expectations: {item.predicted_response_expectations}",
            ])
        return "\n".join(lines)

    def save_state(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in self._items]

    def restore_state(self, state: list[dict[str, object]]) -> None:
        self._items = [Obligation.from_dict(item) for item in state]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)


class TriggerConfig:
    __slots__ = (
        "min_trigger_words",
        "elapsed_seconds",
        "assistant_messages",
        "tool_calls",
        "draft_words_with_work",
        "draft_chars_with_work",
    )

    def __init__(
        self,
        min_trigger_words,
        elapsed_seconds,
        assistant_messages,
        tool_calls,
        draft_words_with_work,
        draft_chars_with_work,
    ):
        self.min_trigger_words = min_trigger_words
        self.elapsed_seconds = elapsed_seconds
        self.assistant_messages = assistant_messages
        self.tool_calls = tool_calls
        self.draft_words_with_work = draft_words_with_work
        self.draft_chars_with_work = draft_chars_with_work


class MemoryRule:
    __slots__ = ("before_elapsed_seconds", "guidance")

    def __init__(self, before_elapsed_seconds, guidance: str):
        self.before_elapsed_seconds = before_elapsed_seconds
        self.guidance = guidance


class PluginConfig:
    __slots__ = (
        "enabled",
        "system_prompt_guidance",
        "triggers",
        "memory_model",
        "rewrite_stages",
        "rewrite_hash",
    )

    def __init__(
        self,
        enabled: bool,
        system_prompt_guidance: str,
        triggers: TriggerConfig,
        memory_model,
        rewrite_stages,
        rewrite_hash: str,
    ):
        self.enabled = enabled
        self.system_prompt_guidance = system_prompt_guidance
        self.triggers = triggers
        self.memory_model = tuple(memory_model)
        self.rewrite_stages = tuple(rewrite_stages)
        self.rewrite_hash = rewrite_hash


def _config_path(*, writable: bool = False) -> Path:
    override = os.environ.get(_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    install_root = os.environ.get("AGENT_ZOO_INSTALL_ROOT", "").strip()
    if install_root:
        data_root = Path(install_root).expanduser()
    else:
        xdg_root = Path(
            os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        ).expanduser()
        data_root = xdg_root / "agent-zoo"
    installed = data_root / "plugin-configs" / "azo-plugin-better-communication" / _CONFIG_NAME
    if writable or installed.exists():
        return installed
    return Path(__file__).resolve().parent.parent / _CONFIG_NAME


def _threshold(value: Any, name: str, *, integer: bool = False):
    label = name if name.startswith("memory_model") else f"triggers.{name}"
    if value is None:
        return None
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative number or null") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative or null")
    return parsed


def _load_config() -> PluginConfig:
    path = _config_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level configuration must be an object")

    system_prompt_guidance = str(raw.get("system_prompt_guidance") or "").strip()
    if not system_prompt_guidance:
        raise ValueError(f"{path}: system_prompt_guidance must be a non-empty string")
    trigger_raw = raw.get("triggers") or {}
    if not isinstance(trigger_raw, dict):
        raise ValueError(f"{path}: triggers must be an object")
    triggers = TriggerConfig(
        min_trigger_words=_threshold(
            trigger_raw.get("min_trigger_words"), "min_trigger_words", integer=True,
        ),
        elapsed_seconds=_threshold(trigger_raw.get("elapsed_seconds"), "elapsed_seconds"),
        assistant_messages=_threshold(
            trigger_raw.get("assistant_messages"), "assistant_messages", integer=True,
        ),
        tool_calls=_threshold(trigger_raw.get("tool_calls"), "tool_calls", integer=True),
        draft_words_with_work=_threshold(
            trigger_raw.get("draft_words_with_work"),
            "draft_words_with_work",
            integer=True,
        ),
        draft_chars_with_work=_threshold(
            trigger_raw.get("draft_chars_with_work"),
            "draft_chars_with_work",
            integer=True,
        ),
    )

    memory_raw = raw.get("memory_model")
    if not isinstance(memory_raw, list) or not memory_raw:
        raise ValueError(f"{path}: memory_model must be a non-empty list")
    memory_model = []
    previous_limit = -1.0
    for index, item in enumerate(memory_raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: memory_model[{index}] must be an object")
        guidance = str(item.get("guidance") or "").strip()
        if not guidance:
            raise ValueError(f"{path}: memory_model[{index}] requires guidance")
        raw_limit = item.get("before_elapsed_seconds")
        if raw_limit is None:
            if index != len(memory_raw) - 1:
                raise ValueError(f"{path}: only the final memory_model rule may use null")
            limit = None
        else:
            limit = _threshold(raw_limit, f"memory_model[{index}].before_elapsed_seconds")
            if limit <= previous_limit:
                raise ValueError(f"{path}: memory_model time limits must increase")
            previous_limit = limit
        memory_model.append(MemoryRule(limit, guidance))
    if memory_model[-1].before_elapsed_seconds is not None:
        raise ValueError(f"{path}: memory_model requires a final null fallback rule")

    stages_raw = raw.get("rewrite_stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError(f"{path}: rewrite_stages must be a non-empty list")
    stages = []
    seen = set()
    for index, item in enumerate(stages_raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: rewrite_stages[{index}] must be an object")
        name = str(item.get("name") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not name:
            raise ValueError(f"{path}: rewrite_stages[{index}] requires name")
        if name in seen:
            raise ValueError(f"{path}: duplicate rewrite stage name {name!r}")
        if not prompt:
            raise ValueError(f"{path}: rewrite stage {name!r} requires prompt")
        seen.add(name)
        stages.append(RewriteStage(name=name, prompt=prompt))

    canonical = {
        "memory_model": [
            {
                "before_elapsed_seconds": rule.before_elapsed_seconds,
                "guidance": rule.guidance,
            }
            for rule in memory_model
        ],
        "rewrite_stages": [
            {"name": stage.name, "prompt": stage.prompt}
            for stage in stages
        ],
    }
    rewrite_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PluginConfig(
        enabled=bool(raw.get("enabled", True)),
        system_prompt_guidance=system_prompt_guidance,
        triggers=triggers,
        memory_model=tuple(memory_model),
        rewrite_stages=tuple(stages),
        rewrite_hash=rewrite_hash,
    )

def _assistant_content(entry: Any) -> str:
    for message in getattr(entry, "messages", ()) or ():
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _latest_user_index(entries: list[Any]) -> int | None:
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if (
            getattr(entry, "role", "") == "user"
            and not getattr(entry, "system_generated", False)
        ):
            return index
    return None


def _ordinary_message_count(state) -> int:
    return sum(
        1
        for entry in (getattr(state, "entries", ()) or ())
        if getattr(entry, "role", "") in {"user", "assistant"}
        and not getattr(entry, "system_generated", False)
    )


class TurnMetrics:
    __slots__ = (
        "elapsed_seconds",
        "assistant_messages",
        "intervening_messages",
        "tool_calls",
        "modified_paths",
        "draft_chars",
        "draft_words",
    )

    def __init__(
        self,
        elapsed_seconds,
        assistant_messages,
        intervening_messages,
        tool_calls,
        modified_paths,
        draft_chars,
        draft_words,
    ):
        self.elapsed_seconds = elapsed_seconds
        self.assistant_messages = assistant_messages
        self.intervening_messages = intervening_messages
        self.tool_calls = tool_calls
        self.modified_paths = tuple(modified_paths)
        self.draft_chars = draft_chars
        self.draft_words = draft_words


TriggerHeuristic = Callable[[TurnMetrics, TriggerConfig], str | None]


def _long_turn(metrics: TurnMetrics, triggers: TriggerConfig) -> str | None:
    threshold = triggers.elapsed_seconds
    if threshold is not None and metrics.elapsed_seconds >= threshold:
        return f"turn elapsed time is {_format_duration(metrics.elapsed_seconds)}"
    return None


def _many_agent_messages(metrics: TurnMetrics, triggers: TriggerConfig) -> str | None:
    threshold = triggers.assistant_messages
    if threshold is not None and metrics.assistant_messages >= threshold:
        return f"turn contains {metrics.assistant_messages} assistant messages"
    return None


def _many_tool_calls(metrics: TurnMetrics, triggers: TriggerConfig) -> str | None:
    threshold = triggers.tool_calls
    if threshold is not None and metrics.tool_calls >= threshold:
        return f"turn contains {metrics.tool_calls} work-related tool calls"
    return None


def _long_proposed_message_with_work(
    metrics: TurnMetrics, triggers: TriggerConfig,
) -> str | None:
    if not (metrics.tool_calls or metrics.intervening_messages or metrics.modified_paths):
        return None
    word_threshold = triggers.draft_words_with_work
    char_threshold = triggers.draft_chars_with_work
    if word_threshold is not None and metrics.draft_words >= word_threshold:
        return f"proposed work summary is {metrics.draft_words} words"
    if char_threshold is not None and metrics.draft_chars >= char_threshold:
        return f"proposed work summary is {metrics.draft_chars} characters"
    return None


TRIGGER_HEURISTICS: tuple[TriggerHeuristic, ...] = (
    _long_turn,
    _many_agent_messages,
    _many_tool_calls,
    _long_proposed_message_with_work,
)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _turn_metrics(state, content: str) -> TurnMetrics:
    entries = list(getattr(state, "entries", ()) or ())
    user_index = _latest_user_index(entries)
    turn_start = user_index + 1 if user_index is not None else 0
    current_index = max(turn_start, len(entries) - 1)
    assistant_messages = sum(
        1 for entry in entries[turn_start:]
        if getattr(entry, "role", "") == "assistant"
    )

    first_reply_index = None
    for index in range(turn_start, current_index):
        entry = entries[index]
        if getattr(entry, "role", "") == "assistant" and _assistant_content(entry).strip():
            first_reply_index = index
            break
    gap_start = first_reply_index + 1 if first_reply_index is not None else turn_start
    intervening_messages = sum(
        1 for entry in entries[gap_start:current_index]
        if getattr(entry, "role", "") == "assistant"
    )

    tool_calls = 0
    modified_paths = set()
    for entry in entries[turn_start:current_index]:
        for call in getattr(entry, "tool_calls", None) or ():
            if getattr(call, "name", "") in _COMMUNICATION_TOOL_NAMES:
                continue
            tool_calls += 1
            for artifact in getattr(call, "artifacts", None) or ():
                path = str(getattr(artifact, "path", "") or "").strip()
                if path and getattr(artifact, "kind", "") == "diff":
                    modified_paths.add(path)

    started_at = float(getattr(state, _TURN_STARTED_AT, time.time()) or time.time())
    return TurnMetrics(
        elapsed_seconds=max(0.0, time.time() - started_at),
        assistant_messages=assistant_messages,
        intervening_messages=intervening_messages,
        tool_calls=tool_calls,
        modified_paths=sorted(modified_paths),
        draft_chars=len(content),
        draft_words=len(content.split()),
    )


def _trigger_reasons(metrics: TurnMetrics, triggers: TriggerConfig) -> tuple[str, ...]:
    minimum = triggers.min_trigger_words
    if minimum is not None and minimum > 0 and metrics.draft_words < minimum:
        return ()
    return tuple(
        reason
        for heuristic in TRIGGER_HEURISTICS
        if (reason := heuristic(metrics, triggers))
    )


def _buffer_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rewrite_record(state, buffer_id: str) -> dict[str, object] | None:
    return (getattr(state, _REWRITES, {}) or {}).get(buffer_id)

def _begin_rewrite(
    state,
    config: PluginConfig,
    buffer_id: str,
    text: str,
    *,
    origin: str,
) -> None:
    setattr(state, _ACTIVE_BUFFER, buffer_id)
    getattr(state, _REWRITES)[buffer_id] = {
        "stage_index": 0,
        "buffer_digest": _buffer_digest(text),
        "rewrite_hash": config.rewrite_hash,
        "closing_obligations": None,
        "origin": origin,
        "previewed": False,
        "started_at": time.time(),
    }


def _delivery_status(state, config: PluginConfig, buffer_id: str) -> tuple[bool, str]:
    if buffer_id != str(getattr(state, _ACTIVE_BUFFER, "") or ""):
        return False, "That buffer is not the active message draft."
    try:
        text = state.buffer_manager.resolve_for_read(state, buffer_id).text
    except (KeyError, ValueError) as exc:
        return False, str(exc)
    rewrite = _rewrite_record(state, buffer_id)
    if not rewrite:
        return False, "No rewrite sequence exists for this draft."
    if rewrite.get("rewrite_hash") != config.rewrite_hash:
        return False, "The configured rewrite stages changed after composition began."
    stage_index = int(rewrite.get("stage_index", 0))
    if stage_index < len(config.rewrite_stages):
        stage = config.rewrite_stages[stage_index]
        return False, f"Rewrite stage {stage_index + 1} ({stage.name}) is still required."
    if rewrite.get("buffer_digest") != _buffer_digest(text):
        return False, "The draft changed after the final rewrite stage was accepted."
    obligations = getattr(state, _OBLIGATIONS)
    missing = [
        obligation_id
        for obligation_id in rewrite.get("closing_obligations", [])
        if obligation_id not in obligations.ids()
    ]
    if missing:
        return False, "Selected obligations are no longer present: " + ", ".join(missing)
    return True, "All rewrite stages are complete and delivery is unlocked."


def _record_new_user_turn(state) -> bool:
    entries = list(getattr(state, "entries", ()) or ())
    if not entries:
        return False
    entry = entries[-1]
    if getattr(entry, "role", "") != "user" or getattr(
        entry, "system_generated", False
    ):
        return False

    entry_index = int(getattr(entry, "index", -1))
    previous_index = int(getattr(state, _TURN_USER_ENTRY_INDEX, -1))
    started_at = float(getattr(state, _TURN_STARTED_AT, 0.0) or 0.0)
    if entry_index == previous_index and started_at > 0.0:
        return False

    setattr(state, _TURN_STARTED_AT, time.time())
    setattr(state, _TURN_USER_ENTRY_INDEX, entry_index)
    active = str(getattr(state, _ACTIVE_BUFFER, "") or "")
    if active:
        getattr(state, _REWRITES).pop(active, None)
    setattr(state, _ACTIVE_BUFFER, "")
    return True


def _entry_is_published(entry: Any) -> bool:
    metadata = getattr(entry, "metadata", {}) or {}
    record = metadata.get(_ENTRY_METADATA_KEY, {}) if isinstance(metadata, dict) else {}
    return bool(isinstance(record, dict) and record.get("published"))


class MessageCompositionGuard:
    """Apply configured triggers, stage drafts, and enforce rewritten delivery."""

    reads = {"entries", "pending_tool_calls", "buffer_manager", "step"}
    writes = {
        "entries",
        "pending_tool_calls",
        "buffer_manager",
        _ACTIVE_BUFFER,
        _OBLIGATIONS,
        _REWRITES,
        _DEFERRED_EAGER_TOOL_NAMES,
    }
    optional_reads = {_TURN_STARTED_AT, _DEFERRED_EAGER_TOOL_NAMES}
    init = {
        _ACTIVE_BUFFER: str,
        _OBLIGATIONS: ObligationList,
        _REWRITES: dict,
    }

    def __init__(self, config: PluginConfig):
        self.config = config

    def __call__(self, state):
        deferred_names = getattr(state, _DEFERRED_EAGER_TOOL_NAMES, None)
        if isinstance(deferred_names, set):
            deferred_names.discard(MESSAGE_USER_TOOL)
        if not state.entries:
            return state
        entry = state.entries[-1]
        role = getattr(entry, "role", "")
        if role != "assistant" or not self.config.enabled:
            return state

        if getattr(entry, "step", None) != getattr(state, "step", None):
            return state
        if _entry_is_published(entry):
            setattr(state, _ACTIVE_BUFFER, "")
            return state
        if getattr(entry, "tool_calls", None):
            return state

        content = _assistant_content(entry)
        metrics = _turn_metrics(state, content)
        reasons = _trigger_reasons(metrics, self.config.triggers)
        if reasons:
            self._stage_text_draft(state, entry, content)
        return state

    def _stage_text_draft(self, state, entry, content: str) -> None:
        buffer_id, _ = state.buffer_manager.create_scratch_buffer(
            content,
            path=f"<message-draft:turn-{getattr(state, 'step', 0)}>",
            turn=getattr(state, "step", 0),
        )
        _begin_rewrite(
            state,
            self.config,
            buffer_id,
            content,
            origin="intercepted",
        )
        raw_call = {
            "id": f"call_compose_{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {
                "name": COMPOSE_TOOL,
                "arguments": json.dumps({"buffer": buffer_id}),
            },
        }
        message = entry.messages[0]
        message["content"] = ""
        message["tool_calls"] = [raw_call]
        call = ToolCall.from_raw(raw_call)
        entry.tool_calls = [call]
        entry.tokens = estimate_tokens(call.raw_args)
        state.pending_tool_calls = [call]


def _first_reply_after(entries: list[Any], user_index: int, current_index: int) -> str:
    for entry in entries[user_index + 1:current_index]:
        if getattr(entry, "role", "") != "assistant":
            continue
        text = _assistant_content(entry).strip()
        if text:
            return text
    return "(The agent began working without a substantive textual reply.)"


def _memory_guidance(metrics: TurnMetrics, config: PluginConfig) -> str:
    for rule in config.memory_model:
        if (
            rule.before_elapsed_seconds is None
            or metrics.elapsed_seconds < rule.before_elapsed_seconds
        ):
            return rule.guidance
    return config.memory_model[-1].guidance


def _modified_paths_text(paths: tuple[str, ...]) -> str:
    if not paths:
        return "none recorded"
    shown = list(paths[:6])
    suffix = f"; +{len(paths) - len(shown)} more" if len(paths) > len(shown) else ""
    return ", ".join(shown) + suffix


def _communication_checks(content: str, metrics: TurnMetrics, config: PluginConfig) -> list[str]:
    checks = [
        f"Configured memory model: {_memory_guidance(metrics, config)}",
        "Do not treat the exact earlier user message or immediate reply as text the human can reread while receiving the proposed message.",
        "Make every conclusion, antecedent, and next step understandable from the proposed message itself. Treat the memory model as a risk estimate, not permission to rely on recall.",
        "State the outcome before supporting implementation details when the outcome is known.",
    ]
    if metrics.intervening_messages:
        checks.append(
            f"The attention gap contained {metrics.intervening_messages} intervening assistant messages; do not assume the human followed or retained them."
        )
    if metrics.tool_calls:
        checks.append(
            f"The agent made {metrics.tool_calls} work-related tool calls during the gap; summarize the consequential work rather than referring to 'the changes' or 'the experiment'."
        )
    if metrics.modified_paths:
        checks.append(
            f"Recorded write artifacts show {len(metrics.modified_paths)} external files changed; name the important files or systems when that matters to the outcome."
        )
    if _trigger_reasons(metrics, config.triggers) and metrics.draft_words < 80:
        checks.append(
            "The turn was substantial but the draft is short; verify that it explains the goal, work performed, result, and remaining limitation."
        )
    table_lines = re.findall(r"(?m)^\s*\|.+\|\s*$", content)
    if len(table_lines) >= 2 or "<table" in content.lower():
        checks.append(
            "The message contains a table. State its principal takeaway in prose so the human does not need to reconstruct the conclusion from cells."
        )
    if len(content) > 4_000:
        checks.append(
            "The message is long. Lead with a compact result summary and keep lower-priority detail visibly subordinate."
        )
    return checks


def _rewrite_prompt(config: PluginConfig, stage_index: int) -> str:
    stage = config.rewrite_stages[stage_index]
    return (
        f"Rewrite stage {stage_index + 1} of {len(config.rewrite_stages)}: "
        f"{stage.name}\n{stage.prompt}"
    )


def _attention_preview(state, buffer_id: str, config: PluginConfig) -> str:
    draft = state.buffer_manager.resolve_for_read(state, buffer_id).text
    metrics = _turn_metrics(state, draft)
    entries = list(getattr(state, "entries", ()) or ())
    current_index = max(0, len(entries) - 1)
    user_index = _latest_user_index(entries)
    if user_index is None:
        user_message = "(No ordinary user message was found.)"
        immediate_reply = "(No immediate reply was found.)"
    else:
        user_message = str(entries[user_index].messages[0].get("content") or "").strip()
        immediate_reply = _first_reply_after(entries, user_index, current_index)

    reasons = _trigger_reasons(metrics, config.triggers)
    duration = _format_duration(metrics.elapsed_seconds)
    memory_guidance = _memory_guidance(metrics, config)
    checks = "\n".join(
        f"- {check}" for check in _communication_checks(draft, metrics, config)
    )
    obligations = getattr(state, _OBLIGATIONS).render(
        current_time=time.time(),
        current_message_count=_ordinary_message_count(state),
    )
    rewrite = _rewrite_record(state, buffer_id)
    if not rewrite:
        raise ValueError("No rewrite sequence exists for the active draft")
    if rewrite.get("origin") == "voluntary":
        reason_text = "composition was requested explicitly by the agent"
    else:
        reason_text = "; ".join(reasons) if reasons else "heuristic interception"
    stage_index = int(rewrite.get("stage_index", 0))
    draft_changed_after_completion = (
        stage_index >= len(config.rewrite_stages)
        and rewrite.get("buffer_digest") != _buffer_digest(draft)
    )
    if draft_changed_after_completion:
        final_index = len(config.rewrite_stages) - 1
        next_prompt = _rewrite_prompt(config, final_index)
        next_action = (
            f"The draft changed after completion. Review the changed draft under the final prompt, "
            f"edit it if needed, then call rewrite_response(buffer=\"{buffer_id}\")."
        )
    elif stage_index < len(config.rewrite_stages):
        next_prompt = _rewrite_prompt(config, stage_index)
        if stage_index == 0:
            next_action = (
                f"Edit {buffer_id}, then call rewrite_response(buffer=\"{buffer_id}\", "
                "closing_obligations=[...]) with the obligation IDs this response will fulfill. "
                "Use an empty list if the message fulfills no recorded obligation."
            )
        else:
            next_action = (
                f"Edit {buffer_id}, then call rewrite_response(buffer=\"{buffer_id}\")."
            )
    else:
        next_prompt = "All configured rewrite stages are complete."
        next_action = (
            f"Call message_user(buffer=\"{buffer_id}\") as the sole tool call. "
            "Any further edit requires the final rewrite stage to be accepted again."
        )

    return (
        f"Draft buffer: {buffer_id}\n"
        f"Composition trigger: {reason_text}\n\n"
        "## Simulated human experience across time\n\n"
        "These stages are temporal, not adjacent context. The exact historical text is shown to you for auditing; do not assume the human can reread or precisely recall it when the final message appears.\n\n"
        "### 1. Turn begins: exact user message recorded in the transcript\n"
        f"{user_message or '(empty user message)'}\n\n"
        "### 2. Agent's immediate reply at the start of work\n"
        f"{immediate_reply}\n\n"
        "### 3. Attention gap: time passes and work occurs\n"
        f"- Elapsed time: {duration}\n"
        f"- Intervening assistant messages after the immediate reply: {metrics.intervening_messages}\n"
        f"- Work-related tool calls: {metrics.tool_calls}\n"
        f"- External files with recorded modifications: {len(metrics.modified_paths)} ({_modified_paths_text(metrics.modified_paths)})\n\n"
        "The human experiences this interval as elapsed time and intermittent transcript activity. Their memory of the original wording and intermediate details may degrade during this gap.\n\n"
        "### 4. Estimated user memory when the new message appears\n"
        f"{memory_guidance}\n\n"
        "### 5. Message the human will see now\n"
        f"{draft or '(empty draft)'}\n\n"
        "## Current obligations\n\n"
        f"{obligations}\n\n"
        "## Communication checks\n\n"
        f"{checks}\n\n"
        "## Required rewrite\n\n"
        f"{next_prompt}\n\n"
        f"{next_action}"
    )


class RecordObligation(Tool):
    """Push a user-facing communication obligation onto the ordered list."""

    name = RECORD_OBLIGATION_TOOL
    group = "Communication"
    reads = {"entries", "step"}
    writes = {_OBLIGATIONS}

    @staticmethod
    def fn(
        user_language: str,
        predicted_response_expectations: str,
        state=None,
    ) -> str:
        """Record a user obligation that must be communicated back later.

        Args:
            user_language: Concise statement of what the user asked, corrected, decided, or expects.
            predicted_response_expectations: What a satisfactory future response should communicate.
        """
        user_language = str(user_language or "").strip()
        expectations = str(predicted_response_expectations or "").strip()
        if not user_language:
            return "ERROR: user_language must not be empty"
        if not expectations:
            return "ERROR: predicted_response_expectations must not be empty"
        entries = list(getattr(state, "entries", ()) or ())
        source_index = _latest_user_index(entries)
        source_message = ""
        if source_index is not None:
            source_message = str(entries[source_index].messages[0].get("content") or "").strip()
        return getattr(state, _OBLIGATIONS).add(
            user_language=user_language,
            predicted_response_expectations=expectations,
            source_user_message=source_message,
            source_entry_index=source_index if source_index is not None else -1,
            message_count=_ordinary_message_count(state),
        )


class ListObligations(Tool):
    """List current obligations with time and message-count ages."""

    name = LIST_OBLIGATIONS_TOOL
    group = "Communication"
    reads = {"entries", _OBLIGATIONS}

    @staticmethod
    def fn(state=None) -> str:
        """List current obligations with time and message-count ages."""
        return getattr(state, _OBLIGATIONS).render(
            current_time=time.time(),
            current_message_count=_ordinary_message_count(state),
        )


class CloseObligation(Tool):
    """Remove an obligation and detach it from any active publication selection."""

    name = CLOSE_OBLIGATION_TOOL
    group = "Communication"
    reads = {_REWRITES}
    writes = {_OBLIGATIONS, _REWRITES}

    @staticmethod
    def fn(obligation_id: str, reason: str, state=None) -> str:
        """Remove an obligation without requiring message publication.

        Use this escape hatch when an obligation was withdrawn, replaced,
        duplicated, recorded incorrectly, or already fulfilled through a short
        response that bypassed the publication protocol. If an active draft had
        selected the obligation, the selection is updated automatically.

        Args:
            obligation_id: Exact obligation identifier returned by record_obligation.
            reason: Why removal without response publication is appropriate.
        """
        obligation_id = str(obligation_id or "").strip()
        reason = str(reason or "").strip()
        if not reason:
            return "ERROR: reason must not be empty"
        removed = getattr(state, _OBLIGATIONS).remove(obligation_id)
        if removed is None:
            return f"ERROR: obligation not found: {obligation_id}"

        detached = 0
        for rewrite in getattr(state, _REWRITES).values():
            selected = rewrite.get("closing_obligations")
            if isinstance(selected, list) and obligation_id in selected:
                rewrite["closing_obligations"] = [
                    item for item in selected if item != obligation_id
                ]
                detached += 1
        suffix = (
            f"\nDetached from {detached} draft selection(s)." if detached else ""
        )
        return (
            f"Removed {obligation_id}: {removed.user_language}\n"
            f"Reason: {reason}{suffix}"
        )


class ComposeMessage(Tool):
    """Preview an editable message buffer and the next required rewrite."""

    group = "Communication"
    reads = {
        "pending_tool_calls",
        "entries",
        "step",
        "buffer_manager",
        _ACTIVE_BUFFER,
        _OBLIGATIONS,
        _REWRITES,
    }
    writes = {_ACTIVE_BUFFER, _REWRITES}
    optional_reads = {_TURN_STARTED_AT}

    @staticmethod
    def fn(buffer: str) -> str:
        """Start or preview composition for an editable message buffer.

        Args:
            buffer: Editable buffer containing the complete proposed message.
        """

    def __init__(self, config: PluginConfig):
        self.config = config

    def available(self, state) -> bool:
        return self.config.enabled

    def transform(self, state):
        for tc, kwargs in self.pending(state):
            try:
                buffer_id = str(kwargs.get("buffer") or "")
                if not buffer_id:
                    raise ValueError("compose_message requires a buffer")
                active = str(getattr(state, _ACTIVE_BUFFER, "") or "")
                if active and buffer_id != active:
                    try:
                        state.buffer_manager.resolve_for_read(state, active)
                    except (KeyError, ValueError):
                        getattr(state, _REWRITES).pop(active, None)
                        setattr(state, _ACTIVE_BUFFER, "")
                        active = ""
                    else:
                        raise ValueError(
                            f"composition is already active for {active}; continue that "
                            "buffer with rewrite_response or message_user"
                        )

                buffer = state.buffer_manager.resolve_for_read(state, buffer_id)
                if getattr(buffer, "readonly", False):
                    raise ValueError("compose_message requires an editable buffer")
                rewrite = _rewrite_record(state, buffer_id)
                if not active or not rewrite:
                    _begin_rewrite(
                        state,
                        self.config,
                        buffer_id,
                        buffer.text,
                        origin="voluntary",
                    )
                    rewrite = _rewrite_record(state, buffer_id)
                elif rewrite.get("rewrite_hash") != self.config.rewrite_hash:
                    origin = str(rewrite.get("origin") or "voluntary")
                    _begin_rewrite(
                        state,
                        self.config,
                        buffer_id,
                        buffer.text,
                        origin=origin,
                    )
                    rewrite = _rewrite_record(state, buffer_id)

                if not bool(rewrite.get("previewed", False)):
                    rewrite["previewed"] = True
                    tc.result = _attention_preview(state, buffer_id, self.config)
                    tc.error = False
                else:
                    stage_index = int(rewrite.get("stage_index", 0))
                    digest_changed = rewrite.get("buffer_digest") != _buffer_digest(
                        buffer.text
                    )
                    if stage_index < len(self.config.rewrite_stages):
                        raise ValueError(
                            f"composition is already active at rewrite stage "
                            f"{stage_index + 1}; edit {buffer_id} and call "
                            f"rewrite_response(buffer=\"{buffer_id}\")"
                        )
                    if digest_changed:
                        raise ValueError(
                            f"the accepted draft changed; call "
                            f"rewrite_response(buffer=\"{buffer_id}\") to reaccept "
                            "the final stage"
                        )
                    raise ValueError(
                        f"all rewrite stages are complete; call "
                        f"message_user(buffer=\"{buffer_id}\") as the sole tool call"
                    )
            except Exception as exc:
                tc.result = f"ERROR: {type(exc).__name__}: {exc}"
                tc.error = True
            self._notify(state, tc)
        return state


class RewriteResponse(Tool):
    """Accept one changed draft and reveal the next configured rewrite stage."""

    name = REWRITE_TOOL
    description = "Submit a changed message draft for the current rewrite stage and receive the next stage prompt."
    group = "Communication"
    reads = {"pending_tool_calls", "buffer_manager", _ACTIVE_BUFFER, _OBLIGATIONS, _REWRITES}
    writes = {_REWRITES}

    schema = {
        "type": "function",
        "function": {
            "name": REWRITE_TOOL,
            "description": "Submit a changed message draft for the current rewrite stage and receive the next stage prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "buffer": {
                        "type": "string",
                        "description": "Active editable message buffer being rewritten.",
                    },
                    "closing_obligations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "On the first stage only, obligation IDs this response intends to fulfill. Use an empty list if none.",
                    },
                },
                "required": ["buffer"],
                "additionalProperties": False,
            },
        },
    }

    @staticmethod
    def fn(
        buffer: str,
        closing_obligations: list[str] | None = None,
    ) -> str:
        """Submit a changed draft for the current rewrite stage.

        The first call must select the obligations the response will fulfill.
        Later calls preserve that selection automatically and should omit
        ``closing_obligations``.

        Args:
            buffer: Active editable message buffer being rewritten.
            closing_obligations: On the first stage only, obligation IDs this response intends to fulfill. Use an empty list if none.
        """

    def __init__(self, config: PluginConfig):
        self.config = config

    def available(self, state) -> bool:
        return self.config.enabled

    def transform(self, state):
        for tc, kwargs in self.pending(state):
            errors = []
            buffer_id = str(kwargs.get("buffer") or "")
            active = str(getattr(state, _ACTIVE_BUFFER, "") or "")
            if not buffer_id or buffer_id != active:
                errors.append("buffer must be the active message draft")
            rewrite = _rewrite_record(state, buffer_id)
            if not rewrite:
                errors.append("no rewrite sequence exists for this buffer")
            elif rewrite.get("rewrite_hash") != self.config.rewrite_hash:
                errors.append("configured rewrite stages changed after composition began")

            try:
                draft = state.buffer_manager.resolve_for_read(state, buffer_id).text
            except (KeyError, ValueError) as exc:
                draft = ""
                errors.append(str(exc))

            stage_index = int(rewrite.get("stage_index", 0)) if rewrite else 0
            digest = _buffer_digest(draft)
            if rewrite:
                if stage_index >= len(self.config.rewrite_stages):
                    if digest == rewrite.get("buffer_digest"):
                        errors.append("all rewrite stages are already complete and the draft is unchanged")
                    else:
                        stage_index = len(self.config.rewrite_stages) - 1
                elif digest == rewrite.get("buffer_digest"):
                    errors.append("the buffer hash is unchanged; revise the draft before advancing")

            supplied = kwargs.get("closing_obligations")
            if stage_index == 0:
                if supplied is None:
                    errors.append("closing_obligations is required on the first rewrite stage")
                    selected = []
                elif not isinstance(supplied, list) or any(not isinstance(item, str) for item in supplied):
                    errors.append("closing_obligations must be a list of obligation IDs")
                    selected = []
                else:
                    selected = [item.strip() for item in supplied if item.strip()]
                    if len(selected) != len(set(selected)):
                        errors.append("closing_obligations contains duplicate IDs")
                    known = set(getattr(state, _OBLIGATIONS).ids())
                    unknown = [item for item in selected if item not in known]
                    if unknown:
                        errors.append("unknown obligation IDs: " + ", ".join(unknown))
            else:
                selected = list(rewrite.get("closing_obligations", [])) if rewrite else []
                if supplied not in (None, []):
                    errors.append(
                        "closing_obligations is accepted only on the first rewrite stage; "
                        "later stages preserve the original selection automatically"
                    )

            if errors:
                tc.result = (
                    "ERROR: " + "; ".join(errors) + ". "
                    + self._next_action(state, buffer_id)
                )
                tc.error = True
            else:
                rewrite["closing_obligations"] = selected
                rewrite["buffer_digest"] = digest
                rewrite["stage_index"] = stage_index + 1
                rewrite["accepted_at"] = time.time()
                if rewrite["stage_index"] >= len(self.config.rewrite_stages):
                    tc.result = (
                        f"Rewrite stage {stage_index + 1} accepted. All configured stages are complete. "
                        f"Submission is unlocked; call message_user(buffer=\"{buffer_id}\") as the sole tool call. "
                        "Any further buffer edit will lock submission again."
                    )
                else:
                    tc.result = (
                        f"Rewrite stage {stage_index + 1} accepted.\n\n"
                        f"{_rewrite_prompt(self.config, rewrite['stage_index'])}\n\n"
                        f"Edit {buffer_id}, then call rewrite_response(buffer=\"{buffer_id}\")."
                    )
            self._notify(state, tc)
        return state


    def _next_action(self, state, buffer_id: str) -> str:
        active = str(getattr(state, _ACTIVE_BUFFER, "") or "")
        if not active:
            return "Put the complete response in an editable buffer and call compose_message(buffer=...)."
        if buffer_id != active:
            return f"Call rewrite_response(buffer=\"{active}\") for the active draft."
        rewrite = _rewrite_record(state, active)
        if not rewrite:
            return f"Call compose_message(buffer=\"{active}\") to initialize its rewrite sequence."
        if rewrite.get("rewrite_hash") != self.config.rewrite_hash:
            return f"Call compose_message(buffer=\"{active}\") to restart the stages under the current rewrite configuration."
        stage_index = int(rewrite.get("stage_index", 0))
        try:
            draft = state.buffer_manager.resolve_for_read(state, active).text
        except (KeyError, ValueError):
            return "Put the response in a valid editable buffer and call compose_message with that buffer."
        if stage_index >= len(self.config.rewrite_stages):
            if rewrite.get("buffer_digest") == _buffer_digest(draft):
                return f"Call message_user(buffer=\"{active}\") as the sole tool call."
            return f"Call rewrite_response(buffer=\"{active}\") to reaccept the changed final draft."
        return f"Edit {active} for rewrite stage {stage_index + 1}, then call rewrite_response again."


class MessageUser(Tool):
    """Publish an accepted message buffer through a visible terminal tool call."""

    group = "Communication"
    reads = {"pending_tool_calls", "buffer_manager", _ACTIVE_BUFFER, _OBLIGATIONS, _REWRITES}
    writes = {_PENDING_PUBLICATIONS}
    init = {_PENDING_PUBLICATIONS: dict}

    @staticmethod
    def fn(buffer: str) -> str:
        """Publish a fully rewritten message buffer to the human.

        A successful call remains visible in the transcript with its tool result.
        The accepted buffer content is then appended as the next assistant message.

        Args:
            buffer: Active buffer whose complete contents should be published.
        """

    def __init__(self, config: PluginConfig):
        self.config = config

    def available(self, state) -> bool:
        return self.config.enabled

    def transform(self, state):
        pending_calls = list(getattr(state, "pending_tool_calls", ()) or ())
        for tc, kwargs in self.pending(state):
            buffer_id = str(kwargs.get("buffer") or "")
            errors = []
            if len(pending_calls) != 1 or pending_calls[0].id != tc.id:
                errors.append(
                    "message_user must be the sole tool call in its assistant response"
                )
            if not buffer_id:
                errors.append("buffer is required")
                current = False
                reason = "No message buffer was supplied."
            else:
                current, reason = _delivery_status(state, self.config, buffer_id)
                if not current:
                    errors.append(reason)

            content = ""
            if current:
                content = state.buffer_manager.resolve_for_read(state, buffer_id).text
                if not content.strip():
                    errors.append("the accepted message buffer is empty")

            if errors:
                tc.result = "ERROR: " + "; ".join(errors) + ". " + self._next_action(
                    state, buffer_id
                )
                tc.error = True
            else:
                rewrite = _rewrite_record(state, buffer_id) or {}
                closing_obligations = list(rewrite.get("closing_obligations", []))
                getattr(state, _PENDING_PUBLICATIONS)[tc.id] = {
                    "buffer_id": buffer_id,
                    "content": content,
                    "closing_obligations": closing_obligations,
                    "accepted_at": time.time(),
                }
                tc.result = (
                    f"Message published successfully from {buffer_id}. "
                    "The accepted buffer content has been added as the next assistant message."
                )
                tc.error = False
            self._notify(state, tc)
        return state

    def _next_action(self, state, buffer_id: str) -> str:
        active = str(getattr(state, _ACTIVE_BUFFER, "") or "")
        if not active:
            return "Put the complete response in an editable buffer and call compose_message(buffer=...)."
        if buffer_id != active:
            return f"The active draft is {active}; continue that buffer or call close_obligation for stale obligations."
        rewrite = _rewrite_record(state, active)
        if not rewrite:
            return f"Call compose_message(buffer=\"{active}\") to initialize its rewrite sequence."
        if rewrite.get("rewrite_hash") != self.config.rewrite_hash:
            return f"Call compose_message(buffer=\"{active}\") to refresh the draft under the current rewrite configuration."
        stage_index = int(rewrite.get("stage_index", 0))
        if stage_index < len(self.config.rewrite_stages):
            return f"Edit {active} and call rewrite_response(buffer=\"{active}\")."
        return f"If the draft changed, call rewrite_response(buffer=\"{active}\"); otherwise call message_user again as the sole tool call."


class AppendPublishedMessage:
    """Append accepted buffer content after the visible message_user tool result."""

    reads = {
        "entries",
        "last_tool_calls",
        _ACTIVE_BUFFER,
        _OBLIGATIONS,
        _REWRITES,
        _PENDING_PUBLICATIONS,
    }
    writes = {
        "entries",
        "last_tool_calls",
        _ACTIVE_BUFFER,
        _OBLIGATIONS,
        _REWRITES,
        _PENDING_PUBLICATIONS,
    }

    def __init__(self, config: PluginConfig):
        self.config = config

    def __call__(self, state):
        publications = getattr(state, _PENDING_PUBLICATIONS)
        appended = False
        for tc in list(getattr(state, "last_tool_calls", ()) or ()):
            payload = publications.pop(tc.id, None)
            if not payload:
                continue
            buffer_id = str(payload["buffer_id"])
            content = str(payload["content"])
            closing_obligations = list(payload.get("closing_obligations", []))
            metadata = {
                _ENTRY_METADATA_KEY: {
                    "published": True,
                    "published_at": time.time(),
                    "publication_tool_call_id": tc.id,
                    "buffer_id": buffer_id,
                    "buffer_digest": _buffer_digest(content),
                    "rewrite_hash": self.config.rewrite_hash,
                    "rewrite_stages": [
                        stage.name for stage in self.config.rewrite_stages
                    ],
                    "closed_obligations": closing_obligations,
                }
            }
            state.entries.append(Entry(
                messages=[{"role": "assistant", "content": content}],
                index=len(state.entries),
                step=getattr(state, "step", 0),
                tokens=estimate_tokens(content),
                metadata=metadata,
            ))
            getattr(state, _OBLIGATIONS).remove_many(closing_obligations)
            getattr(state, _REWRITES).pop(buffer_id, None)
            if str(getattr(state, _ACTIVE_BUFFER, "") or "") == buffer_id:
                setattr(state, _ACTIVE_BUFFER, "")
            appended = True
        if appended:
            state.last_tool_calls = []
        return state


class CommunicationProtocolSystemPrompt(RenderTransformRegistrar):
    """Register configurable communication guidance on the model-channel system prompt."""

    reads = {"entries"}
    writes = {
        _TURN_STARTED_AT,
        _TURN_USER_ENTRY_INDEX,
        _ACTIVE_BUFFER,
        _REWRITES,
    }
    init = {
        _TURN_STARTED_AT: float,
        _TURN_USER_ENTRY_INDEX: int,
    }
    transform_name = "human attention composition system prompt"

    def __init__(self, config: PluginConfig):
        self.guidance = config.system_prompt_guidance

    def __call__(self, state):
        _record_new_user_turn(state)
        if not state.entries:
            return state
        entry = state.entries[0]
        if self.transform_name in entry.render_transform_names(
            MODEL_RENDER_CHANNEL,
            pending=True,
        ):
            return state
        block = "# Message Composition\n" + self.guidance

        def transform(messages, _state, _block=block):
            if not messages:
                return messages
            first = dict(messages[0])
            first["content"] = CommunicationProtocolSystemPrompt._append_content(
                first.get("content"),
                _block,
            )
            return [first, *messages[1:]]

        entry.append_render_transform(
            MODEL_RENDER_CHANNEL,
            transform,
            name=self.transform_name,
        )
        return state

    @staticmethod
    def _append_content(content: Any, block: str) -> Any:
        if isinstance(content, str):
            return content.rstrip() + "\n\n" + block
        if content is None:
            return block
        if isinstance(content, list):
            rendered = list(content)
            uses_input_blocks = any(
                isinstance(item, Mapping)
                and str(item.get("type") or "").startswith("input_")
                for item in rendered
            )
            rendered.append({
                "type": "input_text" if uses_input_blocks else "text",
                "text": block,
            })
            return rendered
        return str(content).rstrip() + "\n\n" + block


def _settings_document(path: Path) -> dict[str, Any]:
    """Read strictly so edits never replace malformed configuration."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SettingError("Create the user plugin configuration before editing defaults") from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SettingError("Cannot read plugin configuration") from None
    if not isinstance(document, dict):
        raise SettingError("Plugin configuration must be an object")
    return document


def _save_enabled(value) -> None:
    temporary = None
    try:
        # The bundled read fallback is never an implicit write target.
        path = _config_path(writable=True).resolve()
        document = _settings_document(path)
        if type(value) is not bool:
            raise SettingError("Expected a boolean")
        document["enabled"] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), mode)
            stream.write(json.dumps(document, indent=2) + "\n")
        os.replace(temporary, path)
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, SettingError):
            raise
        raise SettingError("Cannot save plugin configuration") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _register_settings(builder) -> None:
    builder.add(Feature("better_communication_settings", components=[
        Setting("better_communication.enabled", "Enabled by default", "bool",
                get=lambda ctx: bool(_settings_document(_config_path()).get("enabled", True)),
                set=lambda ctx, value: _save_enabled(value),
                group="Better communication", apply="reload",
                help="Saved default, not current tool status; reload to change loaded communication tools."),
    ], order=[Setting]))


def register_features(builder, *, session, config):
    _register_settings(builder)
    plugin_config = _load_config()
    if not plugin_config.enabled:
        return
    builder.add(Feature(
        name="guarded_message_composition",
        components=[
            MessageCompositionGuard(plugin_config),
            RecordObligation(),
            ListObligations(),
            CloseObligation(),
            ComposeMessage(plugin_config),
            RewriteResponse(plugin_config),
            MessageUser(plugin_config),
            AppendPublishedMessage(plugin_config),
            CommunicationProtocolSystemPrompt(plugin_config),
        ],
        order=[
            AppendInput,
            MessageCompositionGuard,
            TurnCounter,
            ToolDispatchStart,
            Tool,
            ConsolidateToolResults,
            AppendPublishedMessage,
            CompressToolResults,
            SystemPromptSkillList,
            CommunicationProtocolSystemPrompt,
            ExcludeForgotten,
            MessageRenderer,
        ],
    ))
