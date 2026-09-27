# DSH premature-stop investigation

This document records only de-identified technical observations from two exported DSH sessions. Raw session archives are not included.

## Session A

Observed totals:

- 70 assistant responses.
- 45 ended with tool-call finish behavior.
- 25 ended with normal `stop` behavior.
- 24 user messages were exactly `繼續完成任務` after premature completion points.

A recurring pattern was:

1. reasoning says the next action will use a tool,
2. the response ends without a tool call,
3. DSH receives provider-level `stop`,
4. the turn closes as completed,
5. the user has to ask the agent to continue.

This means the DSH log does not support a simple theory that DSH always mistakes a completed reasoning item for a completed response: many reasoning-to-tool transitions in the same session succeeded.

## Session B

The second exported run was shorter and cleaner:

- 6 assistant steps total.
- first 5 ended in tool-call behavior successfully.
- step 6 ended in `stop`.
- the failing step had `outputTokens = 1126` while the configured maximum was far larger, ruling out ordinary output-token exhaustion.
- the failing reasoning explicitly planned to download and inspect a file.
- the final text said it was "Now downloading ...".
- no tool-call block appeared after that text.

The request headers observed in this session carried the same tool set/schema across the successful and failing steps. This weakens a hypothesis that the failure was caused by a schema suddenly changing at the failing step.

## Memory injection clue

The session also contained `fount-memory` messages holding prior runs of a very similar task. Some retrieved traces were incomplete and ended around a "now I will download/verify" transition.

The later run produced a very similar transition immediately before premature stop.

This is a strong reason to test memory injection separately, but it is still correlation, not proof.

## Current root-cause ranking

The investigation currently treats these as separate axes:

1. historical-memory / `fount-memory` trajectory imitation,
2. `xhigh` reasoning and model/provider EOS behavior,
3. vLLM reasoning/tool parser boundaries,
4. DSH continuation policy,
5. chat-template structure.

The stable template is intentionally kept conservative so that future A/B tests change one layer at a time.
