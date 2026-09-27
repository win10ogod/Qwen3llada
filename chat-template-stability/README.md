# Qwen 3.8 Chat Template Stability Archive

這是一份針對 Qwen 3.8 / 相容模型 chat template 的實驗、事故記錄與目前穩定版整理。

目標很單純：**保留每次模板修改的原因、已知錯誤與退役版本，避免再次把已知有問題的做法加回來。**

> 這個目錄記錄的是模板層實驗。它不聲稱能修復所有 provider / serving / framework 的 premature stop 問題。

## 目前建議版本

`templates/stable/chat_template.jinja`

目前穩定版的設計原則：

- thinking 永久開啟，沒有 `enable_thinking=false` 路徑。
- 歷史 reasoning 保留。
- `enable_interleaved_thinking` 預設為 `true`。
- 預設 reasoning effort 為 `medium`。
- 支援 `<|think_xhigh|>` / `<|think_medium|>` / `<|think_low|>` 等控制 tag，渲染前移除 tag。
- tool arguments **只做序列化，不做語義解析、修復或 schema 推測**。
  - mapping：展開為 `<parameter=...>`。
  - string：原樣保留。
- 工具提示避免把「簡短正文」與 tool call 寫成互斥規則。

## 版本分類

| 狀態 | 檔案 | 說明 |
|---|---|---|
| Stable | `templates/stable/chat_template.jinja` | 目前建議使用。已撤掉自製 tool-argument parser。 |
| Baseline | `templates/archive/baseline/original-2026-09-27.jinja` | 最初的 171 行模板，用來做差異基準。 |
| Broken | `templates/archive/broken/v2-k3-json-parser.jinja` | 引入 K3 式 JSON parser 與較強的 tool-flow 約束。退役。 |
| Broken | `templates/archive/broken/v3-tool-parser.jinja` | 加入 control-tag pre-scan 與 anti-stall 提示，但仍保留自製 tool parser。退役。 |
| Broken | `templates/archive/broken/v4-minimal-mapping-only.jinja` | 移除自製 parser，但回到 mapping-only `|items` 路徑，對 OpenAI-compatible JSON-string arguments 不安全。退役。 |

完整差異見 [CHANGELOG.md](CHANGELOG.md)。

## 已知問題

premature stop 目前仍可能發生。兩份 DSH session 的去識別化觀察顯示：

- 舊 session：70 個 assistant response，其中 45 個 `tool-calls`、25 個 `stop`；其中 24 個 `stop` 之後需要使用者再次要求繼續。
- 新 session：前 5 個 step 正常 `tool-calls`，第 6 個在 reasoning 已明確計畫下一個工具操作後，以 `stop` 結束，且沒有 tool-call block。
- 新 session 中，工具 schema 在多個 request header 間保持一致，因此「中途突然換 schema」不是該次中斷的合理解釋。
- DSH `fount-memory` 曾把過去未完成的 agent 軌跡重新注入上下文，這是高度可疑的外部因素，但尚未被證明為唯一根因。
- `xhigh` reasoning / provider EOS / reasoning-tool parser 邊界仍需獨立 A/B 測試。

詳見 [docs/DSH_PREMATURE_STOP.md](docs/DSH_PREMATURE_STOP.md)。

## 模板邊界

這個專案刻意遵守一條規則：

> **Chat template 是 serializer，不是 tool schema repair engine。**

模板可以決定 token / role / reasoning / tool-call 的 wire format，但不應猜測工具參數代表什麼，也不應自行把一套 schema 改成另一套 schema。

這條規則來自一次實際退化：將 K3 的 JSON parser 移植進 Qwen 模板後，工具鏈的可觀察失敗面明顯擴大。雖然不能把所有後續工具錯誤都直接歸因於 parser，但該改動沒有必要，且讓根因分析更困難，因此已完全撤除。

## 驗證

執行：

```bash
python tests/test_template.py
```

測試目前覆蓋：

- 基本 Jinja 編譯與渲染。
- mapping tool arguments。
- raw string tool arguments 原樣保留。
- interleaved thinking 預設開啟。
- 關閉 interleaved thinking 不會重新啟用 `enable_thinking`。
- 穩定版中不存在 K3 式 `jp_*` / `numstr` parser。

## 參考來源

第三方模板不直接複製進本目錄，僅保留連結與設計對照：

- Qwen 官方模型 / chat template
- froggeric / Qwen-Fixed-Chat-Templates
- Kimi K3 chat template（僅作交錯 reasoning 與 tool-history 結構參考）

見 [REFERENCES.md](REFERENCES.md)。

## License

本目錄放在 `Qwen3llada` repository 內，沿用該 repository 的 Apache-2.0 license。模板基於 Qwen 系 chat-template 格式演進；第三方參考模板不在此目錄重新分發。

---

### English summary

This directory archives the evolution of a Qwen 3.8-compatible chat template, including retired regressions and the current stable candidate. The stable template keeps reasoning/interleaving support while deliberately avoiding semantic parsing or repair of tool arguments. Premature-stop behavior is still under investigation and is not claimed to be fully solved at the template layer.
