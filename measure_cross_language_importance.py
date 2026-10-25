
import json
import argparse
import os
import torch
from tqdm import tqdm

from llm_utils import LLM_KN


DATA_FILE = "data/cross_language.activation.jsonl"
CACHE_DIR = os.getenv('HF_HOME', '.cache/huggingface')
os.environ['HF_HOME'] = CACHE_DIR






SPECIFIED_NEURONS = [





































    (46, 7410), (49, 8062), (55, 17163), (49, 14899), (54, 6966), (57, 4649), (50, 11755), (60, 5221), (51, 10670), (51, 2640), (55, 5963), (52, 9921), (59, 8551), (54, 11926), (49, 7808), (57, 75), (52, 5247), (52, 5073), (44, 14484), (50, 12340), (61, 15458), (61, 13844), (51, 8320), (58, 13366), (62, 4821), (59, 6656), (32, 8316), (58, 8881), (61, 7547), (52, 13960), (59, 7330), (44, 9805), (49, 9461), (51, 6370), (55, 11908), (47, 14064), (63, 2870), (52, 11418), (50, 804), (56, 14085), (46, 5291), (48, 6847), (61, 12214), (60, 3877), (56, 2519), (63, 11958), (60, 1909), (47, 1323), (50, 7375), (61, 13910), (49, 2434), (61, 12155), (50, 6902), (55, 16157), (51, 7257), (53, 14064), (57, 8146), (52, 4748), (51, 15906), (62, 396), (55, 12157), (49, 15415), (54, 17025), (43, 11655), (50, 2191), (56, 5456), (53, 13200), (61, 5809), (62, 13813), (55, 728), (58, 8220), (62, 16876), (51, 4013), (49, 7769), (49, 9209), (44, 3770), (48, 15992), (56, 2121), (48, 4298), (60, 489), (55, 12742), (50, 16392), (50, 5711), (50, 1336), (49, 3049), (61, 15855), (48, 11153), (45, 16295), (51, 15981), (62, 535), (51, 4818), (51, 8125), (51, 5190), (49, 843), (59, 15037), (42, 16525), (61, 8127), (58, 2990), (56, 11630), (58, 14693), (63, 17405), (61, 12767), (49, 7707), (41, 13429), (47, 6724), (57, 7412), (58, 7804), (62, 9925), (61, 1265), (60, 1623), (54, 12508), (49, 13131), (49, 12356), (57, 3064), (54, 7669), (54, 7788), (49, 5165), (49, 5467), (51, 8374), (45, 13081), (62, 14775), (56, 5344), (51, 11391), (49, 1293), (56, 1088), (50, 6124), (55, 3978), (43, 11625), (59, 2655), (51, 9423), (54, 11854), (46, 897), (61, 3436), (34, 17129), (49, 8267), (61, 7893), (50, 16780), (51, 9816), (47, 11085), (61, 348), (56, 13376), (49, 11659), (49, 12150), (62, 3922), (55, 8529), (46, 6095), (49, 4294), (59, 369), (48, 13173), (48, 8316), (49, 3092), (52, 4320), (47, 6801), (55, 15201), (62, 7778), (42, 545), (55, 6204), (55, 10881), (54, 13461), (59, 10278), (53, 1478), (53, 14557), (49, 11997), (46, 13624), (62, 12964), (59, 10186), (55, 16906), (37, 5198), (54, 10382), (25, 9809), (50, 12773), (50, 17267), (49, 13821), (49, 15380), (53, 5471), (44, 10551), (63, 11068), (36, 16328), (51, 2575), (58, 9433), (60, 11163), (62, 10715), (62, 4061), (55, 2014), (46, 10380), (53, 5254), (51, 10513), (61, 9740), (58, 9817), (63, 13914), (52, 14130), (61, 13532), (49, 2886), (43, 6872), (53, 11279), (53, 8170), (60, 12831), (59, 11393), (61, 14926), (47, 536), (48, 16341), (52, 1545), (56, 10683), (62, 445), (53, 735), (54, 11360), (60, 1567), (59, 7811), (52, 3604), (61, 2057), (60, 6015), (63, 6277), (63, 10542), (62, 7300), (53, 11135), (50, 89), (48, 2160), (63, 8693), (49, 6186), (48, 4448), (49, 1274), (51, 90), (56, 11149), (58, 14505), (58, 12290), (55, 2583), (63, 6496), (52, 16484), (50, 3417), (51, 8986), (51, 12705), (51, 7436), (60, 10383), (59, 8106), (43, 1498), (51, 2460), (46, 1909), (61, 5058), (51, 2542), (61, 2663), (50, 4841), (43, 634), (47, 2398), (61, 2754), (47, 6126), (60, 4497), (52, 8093), (62, 2673), (57, 2622), (36, 11587), (50, 6610), (56, 6385), (60, 1034), (53, 4531), (51, 5043), (50, 14112), (54, 8676), (50, 6192), (61, 6802), (48, 5700), (61, 7705), (54, 8987), (50, 17129), (52, 10765), (63, 4560), (50, 9819), (61, 2018), (55, 6178), (61, 6411), (52, 7302), (18, 8609), (52, 9709), (49, 2611), (54, 8495), (62, 2440), (55, 16169), (50, 9913), (51, 2357), (53, 11443), (60, 16824), (50, 9266), (53, 11150), (52, 5727), (55, 4850), (49, 3733), (51, 14853)


]



def get_cross_language_device_map():
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpus >= 2:
        print(
            f"Detected {visible_gpus} visible GPUs; "
            "using device_map='balanced' for model parallelism."
        )
        return "balanced"

    print(
        f"Detected {visible_gpus} visible GPU(s); "
        "using the original single-device loading."
    )
    return None


class ActivationInjector:

    def __init__(self, model, transformer_layers, layer_idx, neuron_idx):
        self.model = model
        self.transformer_layers = transformer_layers
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.current_activation = None
        self.hook_handle = None

    def set_activation(self, activation_value):
        self.current_activation = activation_value

    def register_hook(self):
        mlp = self.transformer_layers[self.layer_idx].mlp
        injector = self

        def hook_fn(module, input, output):
            if len(output.shape) == 3 and injector.current_activation is not None:
                activation = injector.current_activation
                if isinstance(activation, torch.Tensor):
                    activation = activation.to(device=output.device, dtype=output.dtype)
                    if activation.numel() != output.size(0):
                        raise ValueError(
                            f"Number of injected values {activation.numel()} does not match batch size "
                            f"{output.size(0)}"
                        )
                    output[:, :, injector.neuron_idx] = activation.reshape(-1, 1)
                else:
                    output[:, :, injector.neuron_idx] = activation
            return output

        self.hook_handle = mlp.up_proj.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def get_model_prediction(perspectiveModel, en_input, options):
    probs = perspectiveModel.getProb_batch(en_input, options)
    predicted = max(probs, key=probs.get)
    return predicted, probs


def evaluate_sample(perspectiveModel, en_input, options, correct_answer):
    predicted, _ = get_model_prediction(perspectiveModel, en_input, options)
    return predicted == correct_answer


def evaluate_qwen35_batch(perspectiveModel, entries):
    if not entries:
        return []
    tokenizer = perspectiveModel.kn.tokenizer
    model = perspectiveModel.kn.model
    device = model.device
    rows = []
    row_metadata = []

    for sample_idx, entry in enumerate(entries):
        prompt_ids = entry.get("_qwen_prompt_ids")
        answer_lists = entry.get("_qwen_answer_ids")
        if prompt_ids is None or answer_lists is None:
            messages = [
                {"role": "system", "content": "You are a useful assistant."},
                {"role": "user", "content": entry["en_input"]},
            ]
            prompt_ids = perspectiveModel._apply_chat_template(
                messages
            ).input_ids[0].tolist()
            answer_lists = [
                tokenizer.encode(option, add_special_tokens=False)
                for option in entry["options"]
            ]
            entry["_qwen_prompt_ids"] = prompt_ids
            entry["_qwen_answer_ids"] = answer_lists
        for answer_ids in answer_lists:
            if not answer_ids:
                raise ValueError(f"Example {entry.get('uuid')} contains an empty option")
            rows.append(prompt_ids + answer_ids[:-1])
            row_metadata.append((sample_idx, len(prompt_ids), answer_ids))

    max_len = max(len(row) for row in rows)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    input_ids = torch.full(
        (len(rows), max_len), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(rows), max_len), dtype=torch.long, device=device
    )
    for row_idx, row in enumerate(rows):
        input_ids[row_idx, :len(row)] = torch.as_tensor(
            row, dtype=torch.long, device=device
        )
        attention_mask[row_idx, :len(row)] = 1

    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask)
        sample_scores = [[] for _ in entries]
        for row_idx, (sample_idx, prompt_len, answer_ids) in enumerate(row_metadata):
            positions = torch.arange(
                prompt_len - 1,
                prompt_len - 1 + len(answer_ids),
                device=device,
            )
            targets = torch.as_tensor(answer_ids, dtype=torch.long, device=device)
            logits = outputs.logits[row_idx, positions, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            selected = log_probs[
                torch.arange(len(answer_ids), device=device), targets
            ]
            sample_scores[sample_idx].append(selected.mean().item())

    results = []
    for entry, scores in zip(entries, sample_scores):
        best_idx = max(range(len(scores)), key=scores.__getitem__)
        results.append(entry["options"][best_idx] == entry["correct_answer"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-language neuron activation-replacement experiment")
    parser.add_argument("--model", "--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--input_file", type=str, default=DATA_FILE)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--qwen_batch_size", type=int, default=4,
        help="Qwen3.5 examples processed in parallel; use 1 or 2 when GPU memory is limited"
    )

    args = parser.parse_args()
    if args.qwen_batch_size <= 0:
        parser.error("--qwen_batch_size must be a positive integer")

    print(f"--- Loading Model: {args.model} ---")
    perspectiveModel = LLM_KN(
        args.model,
        temperature=args.temperature,
        device_map=get_cross_language_device_map(),
    )
    kn = perspectiveModel.kn
    model = kn.model
    transformer_layers = kn._get_transformer_layers()
    is_qwen35 = perspectiveModel._is_qwen35()


    print("Applying performance optimizations...")
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled")

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
        print("Set float32 matmul precision to 'high'")

    print(f"--- Loading Data from: {args.input_file} ---")
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Loaded {len(lines)} records")

    end_idx = args.end if args.end != -1 else len(lines)
    lines = lines[args.start:end_idx]

    print(f"Processing {len(lines)} samples (index {args.start} to {end_idx-1})")
    print(f"Total neurons to analyze: {len(SPECIFIED_NEURONS)}")



    n_layers = len(transformer_layers)
    layer_intermediate_sizes = {}
    for layer_idx, neuron_idx in SPECIFIED_NEURONS:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} is outside the valid range [0, {n_layers - 1}]")
        if layer_idx not in layer_intermediate_sizes:
            layer_intermediate_sizes[layer_idx] = kn.intermediate_size(layer_idx)
        intermediate_size = layer_intermediate_sizes[layer_idx]
        if neuron_idx < 0 or neuron_idx >= intermediate_size:
            raise ValueError(
                f"Layer {layer_idx} neuron index {neuron_idx} is outside the valid range "
                f"[0, {intermediate_size - 1}]"
            )
    print(
        f"Model layers: {n_layers}, Intermediate sizes: "
        f"{sorted(set(layer_intermediate_sizes.values()))}"
    )


    data_entries = []
    for i, line in enumerate(lines):
        entry = json.loads(line.strip())
        uuid = entry.get("uuid")
        en_input = entry.get("en_input", "")
        options = entry.get("options", [])
        correct_answer = entry.get("correct_answer", "")
        activations = entry.get("activations", {})

        if not en_input or not options or not correct_answer:
            print(f"[{i+1}] Warning: Missing data for uuid {uuid}")
            continue
        if not activations:
            print(f"[{i+1}] Warning: No activation data for uuid {uuid}")
            continue

        data_entries.append({
            "uuid": uuid,
            "en_input": en_input,
            "options": options,
            "correct_answer": correct_answer,
            "activations": activations
        })

    print(f"Valid entries: {len(data_entries)}")


    print(f"\n{'='*60}")
    print("Step 1: Calculating Baseline Results")
    print(f"{'='*60}")

    baseline_correct = 0
    total = len(data_entries)

    if is_qwen35:
        for start in tqdm(
            range(0, len(data_entries), args.qwen_batch_size),
            desc="Baseline evaluation",
        ):
            batch_entries = data_entries[start:start + args.qwen_batch_size]
            baseline_correct += sum(evaluate_qwen35_batch(perspectiveModel, batch_entries))
    else:
        for entry in tqdm(data_entries, desc="Baseline evaluation"):
            is_correct = evaluate_sample(perspectiveModel, entry["en_input"], entry["options"], entry["correct_answer"])
            if is_correct:
                baseline_correct += 1

    baseline_accuracy = (baseline_correct / total * 100) if total > 0 else 0
    print(f"Baseline: {baseline_correct}/{total} ({baseline_accuracy:.2f}%)")


    print(f"\n{'='*60}")
    print("Step 2: Per-Neuron Activation Injection")
    print(f"{'='*60}")

    neuron_results = []

    for neuron_idx, neuron_tuple in enumerate(tqdm(SPECIFIED_NEURONS, desc="Neuron ablation")):
        layer = neuron_tuple[0]
        neuron = neuron_tuple[1]
        modified_correct = 0


        sample_activations = []
        for entry in data_entries:
            activations = entry["activations"]
            neuron_key = str((layer, neuron))
            if neuron_key in activations and activations[neuron_key] is not None:
                sample_activations.append((entry, activations[neuron_key]))

        if not sample_activations:
            neuron_results.append({
                "layer": layer,
                "neuron": neuron,
                "modified_correct": 0,
                "modified_accuracy": 0.0,
                "impact": -baseline_correct
            })
            continue


        injector = ActivationInjector(
            model, transformer_layers, layer, neuron
        )
        injector.register_hook()

        if is_qwen35:
            for start in range(0, len(sample_activations), args.qwen_batch_size):
                chunk = sample_activations[start:start + args.qwen_batch_size]
                batch_entries = [item[0] for item in chunk]

                repeated_activations = []
                for entry, activation_value in chunk:
                    repeated_activations.extend([activation_value] * len(entry["options"]))
                injector.set_activation(torch.as_tensor(repeated_activations, device=model.device))
                modified_correct += sum(
                    evaluate_qwen35_batch(perspectiveModel, batch_entries)
                )
        else:
            for entry, activation_value in sample_activations:
                injector.set_activation(activation_value)
                is_correct = evaluate_sample(perspectiveModel, entry["en_input"], entry["options"], entry["correct_answer"])
                if is_correct:
                    modified_correct += 1

        injector.remove_hook()

        modified_accuracy = (modified_correct / total * 100) if total > 0 else 0
        impact = modified_correct - baseline_correct

        neuron_results.append({
            "layer": layer,
            "neuron": neuron,
            "modified_correct": modified_correct,
            "modified_accuracy": modified_accuracy,
            "impact": impact
        })

        if (neuron_idx + 1) % 10 == 0:
            print(f"\n  Progress: {neuron_idx + 1}/{len(SPECIFIED_NEURONS)} neurons processed")


    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Baseline correct: {baseline_correct}/{total} ({baseline_accuracy:.2f}%)")
    print(f"\nNeuron ablation results (ranked by impact):")

    neuron_results_sorted = sorted(neuron_results, key=lambda x: x["impact"], reverse=True)

    print(f"\n{'Layer':<8} {'Neuron':<8} {'Modified':<12} {'Impact':<10}")
    print("-" * 40)
    for result in neuron_results_sorted[:20]:
        print(f"{result['layer']:<8} {result['neuron']:<8} {result['modified_correct']}/{total:<10} {result['impact']:<10}")


    os.makedirs("results/", exist_ok=True)
    output_file = "results/cross_language_importance.json"

    output_data = {
        "summary": {
            "total_samples": total,
            "baseline_correct": baseline_correct,
            "baseline_accuracy": baseline_accuracy,
        },
        "neuron_results": neuron_results_sorted
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete; results saved to: {output_file}")


if __name__ == "__main__":
    main()



