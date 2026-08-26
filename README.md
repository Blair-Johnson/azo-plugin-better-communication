# azo-plugin-better-communication

An experimental Agent Zoo userspace plugin that treats human attention as a guarded resource. It adds an ordered obligation list and a staged rewrite protocol for substantial end-of-turn summaries.

Messages shorter than `min_trigger_words` bypass the rewrite protocol entirely; the default is 40 words. Set it to `0` or `null` to disable this length bypass. For messages at or above the minimum, draft length triggers only when the turn already contains work-related tool calls, intervening assistant messages, or recorded file modifications. Elapsed time, multiple assistant messages, and tool-call count remain independently configurable triggers.

## Obligations

`record_obligation(user_language=..., predicted_response_expectations=...)` pushes an item onto the ordered obligation list and retains the source user message. `list_obligations()` shows every item with its age in elapsed time and transcript messages. The list has no open/closed statuses: fulfilled obligations are removed only after `message_user` successfully publishes a response that selected them for closure.

`close_obligation(obligation_id=..., reason=...)` is an escape hatch that removes an item without publication. It can clear withdrawn, replaced, duplicate, invalid, or stale fulfilled obligations. If a draft selected the obligation for closure, the tool also detaches that selection so the draft remains publishable.

## Rewrite protocol

The entire pipeline is available voluntarily. The agent can place a complete proposed response in any editable buffer and call `compose_message(buffer=...)`; the tool adopts that buffer as the active draft and displays the temporal attention gap, current obligation list, proposed message, and first rewrite prompt. If the agent instead attempts to finish without composing, heuristic interception can create and compose a draft automatically.

The configurable model-channel system-prompt addendum tells agents to record obligations proactively during multi-step work, recommends voluntary composition for substantial handoffs, and recommends skipping composition for simple confirmations with little to report. The plugin does not append this guidance to user messages.

After either entry path, the agent edits the active buffer and calls:

```python
rewrite_response(
    buffer="buf_...",
    closing_obligations=["obl_..."],
)
```

The first call selects the obligations the response intends to fulfill. Later calls use only `rewrite_response(buffer="buf_...")`; the plugin preserves the original obligation selection automatically. Each call requires a buffer hash different from the previous accepted version and returns the next configured rewrite prompt. All communication tools remain visible throughout the lifecycle; calls made at the wrong point return an error that identifies the valid next action. After the final stage, a successful `message_user(buffer=...)` call remains in the transcript with its success result, and the accepted buffer content is appended as the next assistant message. Publication removes the selected obligations atomically.

## Install

```bash
azo-plugin install .
```

Then run `/reload` in the AZO TUI.

## Configuration

Edit the repository `better_communication.json`, reinstall with `azo-plugin install . --force --force-config`, then run `/reload`. `system_prompt_guidance` controls the complete system-prompt addendum. `min_trigger_words` is evaluated before every other heuristic: a positive value bypasses messages with fewer words, while `0` or `null` disables the bypass. Remaining trigger thresholds are OR conditions; set any threshold to `null` to disable it. `draft_words_with_work` and `draft_chars_with_work` never trigger on an immediate no-tool reply. `memory_model` controls elapsed-time recall guidance, and `rewrite_stages` controls the ordered stage names and prompts.

## Prototype boundary

The plugin synthesizes `compose_message` by coordinating the stored assistant message, `Entry.tool_calls`, and `state.pending_tool_calls`, so interception remains experimental. Published messages carry durable metadata on their transcript `Entry`, preventing reload from restaging a message that already passed through the protocol.
