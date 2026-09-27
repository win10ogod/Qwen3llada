# References

These sources informed the investigation. Their full templates are not vendored here.

## Qwen

- Qwen3.8-27B model repository:
  https://huggingface.co/Qwen/Qwen3.8-27B

- Qwen chat-template / tokenizer configuration should be treated as the primary format reference for Qwen-family role and token boundaries.

## froggeric / Qwen-Fixed-Chat-Templates

- Repository:
  https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates

Useful design lesson from this project: even a heavily modified compatibility template can keep raw string tool arguments raw instead of embedding a second semantic JSON repair layer into the chat template.

## Kimi K3

A K3 template was used as a structural reference for interleaved reasoning and tool-history layout.

The K3 JSON parsing machinery is **not** carried into the stable Qwen template. K3 and Qwen have different tool wire formats; copying a parser between them was an unnecessary coupling.

## DeepSeek Harness

- Repository:
  https://github.com/deepseek-ai/deepseek-harness

DSH source was used to verify tool-domain behavior such as the Todo item shape and to separate framework behavior from template behavior.
