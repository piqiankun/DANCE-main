from dance.paths import PROJECT_ROOT

import argparse
import json
import os
import random

from openai import OpenAI


def generate_wrong_options(en_input: str, en_target: str, client: OpenAI, model: str) -> list[str]:
    prompt = (
        f'Given the following English sentence with a BLANK:\n'
        f'"{en_input} ___"\n'
        f'The correct answer is: "{en_target}"\n'
        f'Provide exactly three plausible but incorrect answer options that differ from "{en_target}". '
        'Return one option per line without numbering, bullets, or additional text.'
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=64,
    )
    content = response.choices[0].message.content or ""
    return [line.strip() for line in content.splitlines() if line.strip()][:3]


def process_entry(entry: dict, client: OpenAI, model: str) -> dict:
    target = entry["en_target"]
    wrong_options = generate_wrong_options(entry["en_input"], target, client, model)
    while len(wrong_options) < 3:
        wrong_options.append(f"[extra_option_{len(wrong_options) + 1}]")
    options = [target, *wrong_options[:3]]
    random.shuffle(options)
    result = entry.copy()
    result["options"] = options
    result["correct_answer"] = target
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Create distractor options for cross-language data.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "cross_language_source.jsonl"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "cross_language.jsonl"))
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY before running this script.")
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    with open(args.input, "r", encoding="utf-8") as source:
        entries = [json.loads(line) for line in source if line.strip()]
    results = []
    for index, entry in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] Processing {entry.get('uuid', 'unknown')}")
        try:
            results.append(process_entry(entry, client, args.model))
        except Exception as error:
            print(f"Request failed: {error}")
            results.append(entry)
    output_parent = os.path.dirname(args.output)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as destination:
        for item in results:
            destination.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
