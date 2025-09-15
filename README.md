Qwen3llada
==========

將 Qwen3 架構改為 LLaDA 擴散式遮罩預測（non-causal）推斷/訓練的最小實作，盡量不動原本 Qwen3 的模組：
- 沿用 Qwen3 的層與權重；僅在 `qwen3llada` 中以非自回歸（去除 causal mask）方式前向。
- 依據 `llada/GUIDELINES.md` 的 loss 設計，提供 pre-train 與 SFT 兩種訓練 loss。
- 提供轉換、訓練、推理與分析腳本；整合 HuggingFace datasets。

目錄：
- `modeling/qwen3llada/`: HuggingFace 風格的 Qwen3-LLaDA 定義（`Qwen3LLadaForMaskedLM`）。
- `scripts/train.py`: 訓練迴圈（pretrain/SFT）+ datasets 整合。
- `scripts/inference.py`: 依 LLaDA 提供的 fixed/semi-autoregressive padding 的取樣流程（含 low-confidence/random remasking）。
- `scripts/convert.py`: 由 Qwen3 checkpoint 轉為 qwen3llada 權重排列（不改動權重內容）。
- `scripts/analyze.py`: 對照 Qwen3 與 Qwen3llada 權重鍵值/形狀。
- `scripts/loss.py`: LLaDA 的 forward_process 與 loss 定義。
- `config.yaml`: 範例配置。

核心概念：
- LLaDA = 以 Transformer Encoder（無 causal mask）做 mask token 的同時預測；
- 訓練：按照 GUIDELINES 的 forward_process 對序列加噪，使用 p_mask 權重交叉熵並以批次長度歸一；SFT 不對 prompt 區加噪、以答案長度歸一；
- 推理：每步同時預測所有 [MASK]，依策略 remask 一部分（low-confidence/random），支援 block 方式的 semi-autoregressive padding；
- 權重：直接沿用 Qwen3 的 embedding/層/LM head，僅在前向路徑把 attention mask 設為非因果（None 或自訂 bias）。

快速開始：
1) 轉換權重（可選，訓練可直接從 Qwen3 載入並在記憶體中轉）：
```
python scripts/convert.py --src demo_model/Qwen3-0.6B-Base --dst out/qwen3llada-converted
```
2) 訓練（預訓練模式）：
```
python scripts/train.py --config config.yaml
```
3) 推理：
```
python scripts/inference.py --model out/qwen3llada-base --prompt "What is diffusion LM?" --steps 128 --gen_length 128 --block_length 32 --remasking random_topk
```

注意事項：
- `mask_token_id` 預設使用 126336（與 LLaDA 一致）。若 tokenizer 不包含該 token，僅用於張量替換不影響 forward；
- datasets 預設使用 `wikitext`，實際執行需開網或改成本地資料集；
- 為維持與 Qwen3 架構一致，未修改原 `modeling/qwen3`；`qwen3llada` 僅包裝並將 attention 調為非因果。注意 diffusion 架構不使用 KV-cache，`Qwen3LLadaConfig` 預設 `use_cache=False`，請勿依賴 cache。

訓練時 Block 注意力偏置（可選）：
- `training.block_attention.enabled: true` 可開啟，並設定 `training.block_attention.block_length`。
- 行為：prompt tokens 彼此可見；答案區每個 block 僅能互看（仍可看 prompt）。此訓練配置與半自回歸 padding 取樣更一致，有助穩定與一致性。

測試/驗證：
- `scripts/analyze.py` 可檢查鍵值映射；
- 可在小型隨機設定上做 forward sanity check（見 `test/`）。

授權：沿用各上游原專案授權條款。
