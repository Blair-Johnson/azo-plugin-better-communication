# azo-plugin-better-communication

An experimental Agent Zoo userspace plugin that treats human attention as a guarded resource. It adds an ordered obligation list and a staged rewrite protocol for substantial end-of-turn summaries.

Short immediate replies pass through normally. Draft length triggers only when the turn already contains work-related tool calls, intervening assistant messages, or recorded file modifications. Elapsed time, multiple assistant messages, and tool-call count remain independently configurable triggers.

## Obligations

`record_obligation(user_language=..., predicted_response_expectations=...)` pushes an item onto the ordered obligation list and retains the source user message. `list_obligations()` shows every item with its age in elapsed time and transcript messages. The list has no open/closed statuses: fulfilled obligations are removed only after `message_user` successfully publishes a response that selected them for closure.

`close_obligation(obligation_id=..., reason=...)` removes an item without publication. It is intended for withdrawn, replaced, duplicate, or invalid obligations rather than normal fulfillment.

## Rewrite protocol

When interception occurs, `compose_message(buffer=...)` displays the temporal attention gap, current obligation list, proposed message, and first rewrite prompt. The agent edits the scratch buffer and calls:

```python
rewrite_response(
    buffer="buf_...",
    closing_obligations=["obl_..."],
)
```

The first call selects the obligations the response intends to fulfill. Each call requires a buffer hash different from the previous accepted version and returns the next configured rewrite prompt. After the final stage, `message_user(buffer=...)` becomes available. Successful publication removes the selected obligations atomically.

## Install

```bash
azo-plugin install .
```

Then run `/reload` in the AZO TUI.

## Configuration

Edit `~/.local/share/agent-zoo/plugin-configs/azo-plugin-better-communication/better_communication.json`, then run `/reload`. Trigger thresholds are OR conditions; set any threshold to `null` to disable it. `draft_words_with_work` and `draft_chars_with_work` never trigger on an immediate no-tool reply. `memory_model` controls elapsed-time recall guidance, and `rewrite_stages` controls the ordered stage names and prompts.

## Prototype boundary

The plugin synthesizes `compose_message` by coordinating the stored assistant message, `Entry.tool_calls`, and `state.pending_tool_calls`, so interception remains experimental. Published messages carry durable metadata on their transcript `Entry`, preventing reload from restaging a message that already passed through the protocol.
