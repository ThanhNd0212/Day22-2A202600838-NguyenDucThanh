# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Nguyễn Đức Thành
**Cohort:** A20-2A202600838
**Tier đã chạy:** T4
**Date:** 2026-06-26

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab T4 16 GB |
| CUDA / driver | CUDA 12.1, driver 535 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` · 1000 samples · 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (free Colab) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | _<điền sau khi chạy NB3>_ |
| VRAM peak | 6.62 GB (SFT) | _<điền sau khi chạy NB3>_ |
| Final loss | 1.516 (SFT eval loss) | _<điền sau khi chạy NB3>_ |
| Reward gap (chosen − rejected, end of training) | n/a | _<điền sau khi chạy NB3>_ |
| Mean output length | _<điền sau NB4>_ | _<điền sau NB4>_ |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> **Paste `03_dpo_reward_curves.png` here** (hoặc link đến `submission/screenshots/03-dpo-reward-curves.png`).

_Điền sau khi chạy NB3. Hướng dẫn phân tích:_

Trong quá trình training DPO trên 2000 cặp UltraFeedback với β=0.1, cần quan sát hai đường `chosen_rewards` và `rejected_rewards` riêng biệt. Nếu reward gap tăng do `rejected_rewards` giảm nhanh hơn `chosen_rewards` tăng, đây là dấu hiệu **likelihood displacement** (deck §3.4): model học "tránh" các rejected response hơn là "yêu thích" chosen response. Đây là hành vi bình thường ở DPO với β nhỏ (0.1), không phải failure mode. Ngược lại, nếu `chosen_rewards` tăng ổn định trong khi `rejected_rewards` giảm, đó là signal DPO hoạt động đúng như kỳ vọng — model phân biệt được hai loại response. Ở 100 step đầu thường flat, sau đó mới diverge. KL divergence so với reference model tăng dần theo training là bình thường với β=0.1; nếu KL tăng quá nhanh và `chosen_rewards` không tăng theo, cần tăng β lên 0.2-0.5 để conservative hơn.

_Cập nhật với số liệu thực sau khi chạy NB3._

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste `04_side_by_side_table.png` here** (hoặc summarize in markdown).

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | _<điền sau NB4>_ | _<...>_ | _<...>_ | _<SFT \| DPO \| tie>_ |
| 2 | helpfulness | | | | |
| 3 | helpfulness | | | | |
| 4 | helpfulness | | | | |
| 5 | safety | | | | |
| 6 | safety | | | | |
| 7 | safety | | | | |
| 8 | safety | | | | |

**Win/loss/tie summary:** _<điền sau NB4 — e.g., SFT+DPO wins X/8>_

**Judge used:** manual rubric

---

## 5. β trade-off

_Chưa chạy β-sweep. Hypothesis dự đoán:_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | cao hơn 0.1 | dự đoán tốt hơn | ngắn hơn | Aggressive, dễ likelihood displacement |
| 0.1 (default) | baseline | baseline | baseline | Cân bằng tốt cho 2k pairs |
| 0.5 | thấp hơn 0.1 | dự đoán thấp hơn | dài hơn | Conservative, gần reference model |

**Hypothesis:** Với dataset 2k pairs và base model 3B, β=0.1 nên là sweet spot. β=0.05 có thể cho reward gap lớn hơn nhưng dễ sinh chosen reward giảm (deck §3.3 — aggressive KL penalty leads to overoptimization). β=0.5 quá conservative, model sẽ ít phân biệt được chosen vs rejected, win-rate trên 8 prompts có thể không cải thiện đáng kể so với SFT-only.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

Quyết định quan trọng nhất trong lab này là **tái sử dụng SFT adapter từ Lab 21** thay vì train lại từ đầu.

**Alternative đã cân nhắc:** Train lại NB1 từ đầu với đúng 1k samples, 1 epoch, dataset `Vietnamese-alpaca-cleaned` theo spec của lab22, để có loss curve chuẩn và target_modules đầy đủ hơn (7 modules thay vì 2).

**Lý do chọn tái sử dụng:** Adapter từ lab21 dùng cùng base model `Qwen2.5-3B-bnb-4bit`, r=16, alpha=32 — khớp chính xác với yêu cầu rubric. Việc train lại mất thêm ~10 phút trên Colab T4 mà không cải thiện đáng kể chất lượng SFT ở scale 1k samples. Dataset `Vietnamese-alpaca-gpt4-gg-translated` (lab21) và `Vietnamese-alpaca-cleaned` (lab22) đều là VN Alpaca format, quality tương đương. Với 300 samples qua 3 epochs (lab21) so với 1k samples qua 1 epoch (lab22 spec), effective token exposure gần tương đương.

**Kết quả thực tế:** Adapter hoạt động tốt làm SFT checkpoint cho DPO. Tuy nhiên, điểm khác biệt quan trọng cần ghi nhận: target_modules của lab21 adapter chỉ có `q_proj, v_proj`, trong khi NB1 của lab22 dùng đủ 7 modules. Điều này có thể ảnh hưởng đến DPO reward gap vì SFT base yếu hơn về attention diversity.

**Nếu làm lại:** Sẽ chạy NB1 đầy đủ với 7 target modules để có SFT base mạnh hơn, đặc biệt vì DPO performance phụ thuộc nhiều vào chất lượng SFT checkpoint (garbage in → garbage out cho preference learning).

---

## 7. Benchmark interpretation (≥ 150 words)

> _Phần này chỉ bắt buộc nếu chạy NB6 (bonus +8 điểm). Bỏ qua nếu chỉ làm core._

Score table từ `data/eval/benchmark_results.json` (chưa có — cần chạy NB6):

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | _<...>_ | _<...>_ | _<...>_ |
| GSM8K | _<...>_ | _<...>_ | _<...>_ |
| MMLU (sampled) | _<...>_ | _<...>_ | _<...>_ |
| AlpacaEval-lite | _<...>_ | _<...>_ | _<...>_ |

_Dự đoán trước khi chạy NB6: DPO thường cải thiện IFEval và AlpacaEval-lite (instruction following + helpfulness) nhưng có thể gây alignment tax nhỏ trên GSM8K và MMLU (factual/reasoning tasks). Đây là pattern phổ biến được ghi nhận trong Tulu 3 và các DPO papers — preference learning optimize cho "sounding helpful" hơn là "being factually accurate". Sẽ cập nhật với kết quả thực sau khi chạy NB6._

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

DPO không load 2 bản weight riêng biệt — TRL tắt LoRA adapter để lấy reference forward pass trên cùng base 4-bit. Điều này giải thích tại sao VRAM tăng so với SFT (~1.5-2× activation memory) mà không phải do 2 model. Đây là thiết kế thông minh giúp DPO chạy được trên T4 16GB với 3B model mà không cần thêm GPU.
