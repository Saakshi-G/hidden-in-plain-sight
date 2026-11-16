
from datasets import load_dataset


def prepare_cyberseceval():
    # Load the CyberSecEval 3 Visual Prompt Injection benchmark
    dataset = load_dataset(
        "facebook/cyberseceval3-visual-prompt-injection"
    )

    df = dataset["test"].to_pandas()

    # Keep attacks that use instructions embedded in images
    embedded = df[
        df["injection_technique"]
        .astype(str)
        .str.contains("embedded_text_instructions")
    ].copy()

    # Keep the plain embedded-text attacks.
    # Obfuscated attacks will be evaluated separately later.
    plain = embedded[
        ~embedded["injection_technique"]
        .astype(str)
        .str.contains("obfuscated")
    ].copy()

    # Remove secret/password/token leakage attacks.
    # Keep semantic-redirection attacks where success corresponds
    # to a meaningful alternative answer.
    semantic = plain[
        ~plain["judge_question"]
        .str.contains("contain", case=False, na=False)
    ].copy()

    semantic["source"] = "CyberSecEval3"
    semantic["benchmark_type"] = "semantic_redirection"

    semantic.to_json(
        "cyberseceval_semantic_candidates.json",
        orient="records",
        indent=2
    )

    print("Total CyberSecEval cases:", len(df))
    print("Embedded-text cases:", len(embedded))
    print("Plain embedded-text cases:", len(plain))
    print("Semantic-redirection candidates:", len(semantic))


if __name__ == "__main__":
    prepare_cyberseceval()
