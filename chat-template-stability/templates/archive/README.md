# Archive

Archived templates are intentionally kept immutable for regression analysis.

- `baseline/` contains the initial reference template.
- `broken/` contains experiments that should not be promoted back to stable without a new, isolated justification and regression tests.

Do not copy features from `broken/` merely because they look more sophisticated. Several regressions came from adding logic that belonged in the provider/framework layer rather than in Jinja.
