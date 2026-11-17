# 🔍 Hidden in Plain Sight

### Can prompt injection be turned from an attack into a detector?

**Detecting AI agents in online surveys through textual, visual, and
steganographic prompt injection.**

Prompt injection is normally treated as an attack on AI systems. We
explore whether the same behavior can instead be used to detect AI
agents completing online surveys. The idea is simple: place a controlled
instruction in survey content and test whether an AI agent follows it
while a human respondent continues answering the survey normally.

## Injection Methods

### Textual Injection

Our textual baseline adapts the survey prompt-injection approach of Wang
et al. The same attack is inserted at the beginning, middle, or end of
four survey scenarios, with a predefined target response used to measure
whether the injection was followed.

### Visual Injection

Visual instructions are embedded directly into survey images. We test
both clearly visible and lower-perceptibility instructions, allowing us
to study the trade-off between instructions that a vision-language model
can read and those that remain unobtrusive to human respondents.

The visual experiments include synthetic images, an MLLM-IPI baseline,
and a CyberSecEval 3 subset containing 97 semantic-redirection attacks,
from which a 15-case pilot was prepared using the official adversarial
images.

### Steganographic Injection

The steganographic approach uses LSB and DCT-based embedding to conceal
information inside survey images with minimal visible distortion. The
implementation also supports authenticated, image-bound payloads for
distinguishing valid instructions from forged ones.

## Results

### Qwen2.5-VL-7B

The main synthetic evaluation contains 1,200 cases and was run on ASU
Sol using Qwen2.5-VL-7B.

| Condition | Cases | Injection Followed | Rate |
|---|---:|---:|---:|
| Clean | 400 | 0 | **0%** |
| Authenticated | 400 | 136 | **34%** |
| Forged | 400 | 136 | **34%** |

**TPR: 34% · FPR: 0% · Precision: 100% · Accuracy: 56%**

![Qwen2.5-VL evaluation results](./docs/images/qwen_metrics.png)

The model followed 34% of the injected instructions while producing no
positive detections on the 400 clean examples. Authenticated and forged
cases currently use the same injected image at inference time, so their
identical success rates are expected. Authentication-aware behavior has
not yet been evaluated.

### Textual Baseline

The Wang baseline contains four clean prompts and twelve injected
prompts, with attacks placed at the beginning, middle, and end.

All 12 injected prompts produced the target response, while none of the
four clean controls did.

![Wang textual baseline](./docs/images/wang_baseline.png)

This provides an initial textual baseline for comparison with the other
injection channels. The clean and injected prompts currently differ in
their output-format instruction, so the prompt formats still need to be
standardized before treating this as a fully controlled comparison.

### Steganographic Results

The LSB and DCT channels were evaluated for payload capacity, image distortion, and robustness.

| Payload | Channel | PSNR | PNG Recovery |
|---|---|---:|---:|
| 6-character canary | LSB + DCT | 47.6 dB | **100%** |
| 39-character instruction | LSB + DCT | 46.7 dB | **100%** |
| 216-character authenticated token | LSB only | 71.5 dB | **100%** |

Short prompt-injection payloads were recovered exactly from unchanged PNG images. The larger authenticated token exceeded the available DCT capacity but was successfully carried through the LSB channel with very low visible distortion.

The current channels are sensitive to image processing: none of the tested payloads survived JPEG recompression or light Gaussian blur. The authentication layer was also tested independently, with valid tokens accepted and incorrectly signed, expired, and image-transplanted tokens rejected. An end-to-end whitened authenticated token was successfully embedded, extracted, and verified at **69.3 dB PSNR**.

### Visual Baselines

We use external visual prompt-injection baselines alongside our synthetic faint-text approach to test different styles of image-based attacks.

**MLLM-IPI.** A survey-style visual injection was tested with Qwen2.5-VL-3B. The model answered the clean question correctly and continued to give the correct answer after a visible instruction attempted to redirect it, giving a preliminary negative result.

**CyberSecEval 3.** From 1,000 benchmark cases, we identified 400 embedded-text attacks and selected 97 semantic-redirection cases relevant to our setting. CyberSecEval cases were first tested through the MLLM-IPI pipeline, where the attacks showed partial success in redirecting model responses. A 15-case pilot using the official adversarial images has now been adapted to our main visual-injection pipeline for evaluation with Qwen2.5-VL.

## Team

### ☕ Chai of Thought

**Arizona State University · CSE 576**

-   Saakshi Gupta
-   Adwaita Basu Bal
-   Harshinee Nadiminty
-   Neal Chandra

## References

-   Wang et al. (2025). *Safeguarding Crowdsourcing Surveys from ChatGPT
    through Prompt Injection.*
-   Xu and Malkin (2026). *A Penny for Your Prompts: Experiments
    Detecting and Mitigating LLM Usage by Survey Respondents.*
-   AI at Meta (2024). *CyberSecEval 3: Visual Prompt Injection
    Dataset.*
-   Pathade (2025). *Invisible Injections: Exploiting Vision-Language
    Models Through Steganographic Prompt Embedding.*
-   Wang et al. (2025). *WebInject: Prompt Injection Attack to Web
    Agents.*