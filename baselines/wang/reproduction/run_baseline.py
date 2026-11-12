import json
from pathlib import Path

from inject import inject_prompt

OUTPUT_INSTRUCTION = (
    'Only provide answer in JSON format '
    '{"answer": choice} without markdown wrapper.'
)


# -------------------------------
# Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "wang_scenarios.json"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# -------------------------------
# Load Wang scenarios
# -------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    scenarios = json.load(f)


# -------------------------------
# Experimental conditions
# -------------------------------

positions = [
    "begin",
    "middle",
    "end"
]

experiment_cases = []


# -------------------------------
# Generate CLEAN questions
# -------------------------------

for scenario in scenarios:

    experiment_cases.append({
        "scenario": scenario["id"],
        "condition": "clean",
        "position": None,
        "target": scenario["target"],
        "prompt": (
        scenario["question"]
        + "\n"
        + OUTPUT_INSTRUCTION
    )
    })


# -------------------------------
# Generate INJECTED questions
# -------------------------------

for scenario in scenarios:

    for position in positions:

        injected_question = inject_prompt(
            scenario["question"],
            position
        )

        experiment_cases.append({
            "scenario": scenario["id"],
            "condition": "manual_injection",
            "position": position,
            "target": scenario["target"],
            "prompt": injected_question
        })


# -------------------------------
# Save experiment cases
# -------------------------------

output_file = RESULTS_DIR / "generated_prompts.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        experiment_cases,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"Generated {len(experiment_cases)} experiment cases."
)

print(
    f"Saved to: {output_file}"
)