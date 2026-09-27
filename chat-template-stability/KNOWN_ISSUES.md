# Known issues

## 1. Premature stop during agentic tool use

Status: **unresolved / external factors strongly suspected**

Observed shape:

1. Assistant reasoning explicitly identifies the next tool action.
2. Assistant may emit a short commentary line such as "Now downloading ...".
3. No tool-call block follows.
4. Provider response ends with `finish reason = stop` / `stopReason = stop` despite a very large remaining output-token budget.

The stable template is deliberately frozen while this is investigated so that template churn does not contaminate A/B tests.

Likely test axes:

- `fount-memory` ON vs OFF.
- reasoning effort `xhigh` vs `medium`.
- raw provider token/EOS behavior.
- vLLM reasoning/tool parser behavior.
- DSH continuation handling.

## 2. Historical-memory imitation

Status: **highly suspicious, not proven causal**

A DSH session contained a `fount-memory` injection with earlier incomplete agent traces. A later run generated a semantically very similar "next I will download..." transition and stopped at the same stage.

This creates an in-context imitation risk: an incomplete historical trajectory can look like a valid demonstration of how an assistant turn ends.

The raw session archives are intentionally not published here because they contain user/session/environment content.

## 3. Tool schema mismatch (`activeForm` / missing `status`)

Status: **separate from premature stop**

Observed DSH validation error included:

- missing required `todos[i].status`
- undeclared `todos[i].activeForm`

DSH's Todo item schema uses `content` plus `status` (`pending`, `in_progress`, `completed`) and does not declare `activeForm`.

This incident is one reason the template no longer attempts to parse, repair, or semantically normalize tool arguments.

## 4. `noop` tool error

Status: **separate / not attributed to the chat template**

An observed error was `Not yet implemented: noop`.

It is tracked separately because it does not establish that Jinja serialization was the cause. Mixing it into the premature-stop fix would obscure the failure boundary.
