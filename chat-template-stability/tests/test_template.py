from pathlib import Path

from jinja2 import Environment


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "templates" / "stable" / "chat_template.jinja"


def raise_exception(message):
    raise RuntimeError(message)


def render(**kwargs):
    env = Environment()
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))
    defaults = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "add_generation_prompt": True,
    }
    defaults.update(kwargs)
    return template.render(**defaults)


def test_basic_render():
    out = render()
    assert "<|im_start|>user\nhello<|im_end|>" in out
    assert out.endswith("<|im_start|>assistant\n<think>\n")


def test_mapping_tool_arguments_are_serialized_as_parameters():
    out = render(
        messages=[
            {"role": "user", "content": "use tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "demo", "arguments": {"x": 1, "label": "a"}}}
                ],
            },
        ],
        add_generation_prompt=False,
    )
    assert "<function=demo>" in out
    assert "<parameter=x>\n1\n</parameter>" in out
    assert "<parameter=label>\na\n</parameter>" in out


def test_string_tool_arguments_are_preserved_verbatim():
    raw = '{"todos":[{"content":"x","status":"pending"}]}'
    out = render(
        messages=[
            {"role": "user", "content": "use tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "todo", "arguments": raw}}],
            },
        ],
        add_generation_prompt=False,
    )
    assert raw in out


def test_interleaved_thinking_defaults_on_after_tool_result():
    out = render(
        messages=[
            {"role": "user", "content": "use tool"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "demo", "arguments": {}}}]},
            {"role": "tool", "content": "ok"},
        ],
        add_generation_prompt=True,
    )
    assert out.endswith("<|im_start|>assistant\n<think>\n")


def test_interleaved_thinking_can_be_disabled_without_enable_thinking_switch():
    out = render(
        messages=[
            {"role": "user", "content": "use tool"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "demo", "arguments": {}}}]},
            {"role": "tool", "content": "ok"},
        ],
        add_generation_prompt=True,
        enable_interleaved_thinking=False,
        enable_thinking=False,
    )
    assert out.endswith("<|im_start|>assistant\n")
    assert "<think>\n\n</think>" not in out


def test_no_custom_json_parser_helpers_remain():
    source = TEMPLATE_PATH.read_text(encoding="utf-8")
    forbidden = ["macro jp_", "numstr(", "jp_value(", "jp_object(", "jp_array("]
    for marker in forbidden:
        assert marker not in source


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
