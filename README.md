# Hidden in Plain Sight

**Detecting AI agents in online surveys using textual, visual, and steganographic prompt injection.**

This project investigates whether adversarial prompt-injection techniques can be repurposed to detect AI agents completing online surveys. We compare textual, visual, and steganographic prompt injection methods and evaluate their effectiveness across different language and vision-language models.

## Approach

Our experimental pipeline evaluates both clean and injected versions of survey questions:

Survey Question
       ↓
Clean / Injected Variant
       ↓
LLM or Vision-Language Agent
       ↓
Model Response
       ↓
Injection Evaluation
       ↓
Detection Result

## We investigate three primary approaches:

Textual Prompt Injection — controlled instructions embedded in survey text.

Visual Prompt Injection — instructions delivered through survey images.

Steganographic Prompt Injection — low-perceptibility instructions concealed within visual survey content.

## Evaluation

We evaluate:

Injection success / intended instruction compliance

AI-agent detection performance

Cross-model transferability

Human perceptibility and false positives

Robustness against injection-aware agents

## Repository Structure

hidden-in-plain-sight/
├── agents/                 # LLM/VLM survey agents
├── attacks/
│   ├── textual/
│   ├── visual/
│   └── steganographic/
├── survey/                 # Survey questions and variants
├── datasets/               # Dataset preparation/loading
├── evaluation/             # Evaluation metrics
├── experiments/            # Experiment scripts/configurations
├── results/                # Experimental results
├── scripts/                # Utility scripts
├── requirements.txt
└── README.md

## Team

Chai of Thought

Saakshi Gupta

Adwaita Basu Bal

Harshinee Nadiminty

Neal Chandra

## Status

Work in progress.
