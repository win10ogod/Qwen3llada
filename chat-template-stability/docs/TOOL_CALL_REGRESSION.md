# Tool-call regression: why the custom parser was removed

## The mistake

An experimental version imported a K3-style JSON parser into the Qwen Jinja template.

The motivation was to accept OpenAI-style `function.arguments` strings and reconstruct Qwen XML `<parameter=...>` blocks.

That was the wrong abstraction boundary.

## Why it was removed

A chat template should serialize the message structure it receives. It should not try to infer the semantic meaning of tool arguments or repair them to match a tool schema.

Every additional parser layer creates another place where:

- types can change,
- escaping can change,
- duplicate keys can be reinterpreted,
- nested values can be reformatted,
- an upstream/provider contract can be silently rewritten.

The stable version therefore follows this rule:

```text
mapping arguments -> serialize parameters
string arguments  -> preserve verbatim
```

No semantic correction is attempted.

## Observed schema error

During the same experimental period DSH reported Todo validation errors involving undeclared `activeForm` fields and missing required `status` fields.

That error does not prove the Jinja parser invented `activeForm`; the model may simply have produced the wrong schema. The parser was removed because it was unnecessary and made attribution harder, not because a direct causal path was proven.
