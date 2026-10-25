from dance.paths import DATA_DIR as PROJECT_DATA_DIR, PROMPT_DIR as PROJECT_PROMPT_DIR, RESULTS_DIR as PROJECT_RESULTS_DIR

SPECIFIED_NEURONS = [



































    (46, 7410), (49, 8062), (55, 17163), (49, 14899), (54, 6966), (57, 4649), (50, 11755), (60, 5221), (51, 10670), (51, 2640), (55, 5963), (52, 9921), (59, 8551), (54, 11926), (49, 7808), (57, 75), (52, 5247), (52, 5073), (44, 14484), (50, 12340), (61, 15458), (61, 13844), (51, 8320), (58, 13366), (62, 4821), (59, 6656), (32, 8316), (58, 8881), (61, 7547), (52, 13960), (59, 7330), (44, 9805), (49, 9461), (51, 6370), (55, 11908), (47, 14064), (63, 2870), (52, 11418), (50, 804), (56, 14085), (46, 5291), (48, 6847), (61, 12214), (60, 3877), (56, 2519), (63, 11958), (60, 1909), (47, 1323), (50, 7375), (61, 13910), (49, 2434), (61, 12155), (50, 6902), (55, 16157), (51, 7257), (53, 14064), (57, 8146), (52, 4748), (51, 15906), (62, 396), (55, 12157), (49, 15415), (54, 17025), (43, 11655), (50, 2191), (56, 5456), (53, 13200), (61, 5809), (62, 13813), (55, 728), (58, 8220), (62, 16876), (51, 4013), (49, 7769), (49, 9209), (44, 3770), (48, 15992), (56, 2121), (48, 4298), (60, 489), (55, 12742), (50, 16392), (50, 5711), (50, 1336), (49, 3049), (61, 15855), (48, 11153), (45, 16295), (51, 15981), (62, 535), (51, 4818), (51, 8125), (51, 5190), (49, 843), (59, 15037), (42, 16525), (61, 8127), (58, 2990), (56, 11630), (58, 14693), (63, 17405), (61, 12767), (49, 7707), (41, 13429), (47, 6724), (57, 7412), (58, 7804), (62, 9925), (61, 1265), (60, 1623), (54, 12508), (49, 13131), (49, 12356), (57, 3064), (54, 7669), (54, 7788), (49, 5165), (49, 5467), (51, 8374), (45, 13081), (62, 14775), (56, 5344), (51, 11391), (49, 1293), (56, 1088), (50, 6124), (55, 3978), (43, 11625), (59, 2655), (51, 9423), (54, 11854), (46, 897), (61, 3436), (34, 17129), (49, 8267), (61, 7893), (50, 16780), (51, 9816), (47, 11085), (61, 348), (56, 13376), (49, 11659), (49, 12150), (62, 3922), (55, 8529), (46, 6095), (49, 4294), (59, 369), (48, 13173), (48, 8316), (49, 3092), (52, 4320), (47, 6801), (55, 15201), (62, 7778), (42, 545), (55, 6204), (55, 10881), (54, 13461), (59, 10278), (53, 1478), (53, 14557), (49, 11997), (46, 13624), (62, 12964), (59, 10186), (55, 16906), (37, 5198), (54, 10382), (25, 9809), (50, 12773), (50, 17267), (49, 13821), (49, 15380), (53, 5471), (44, 10551), (63, 11068), (36, 16328), (51, 2575), (58, 9433), (60, 11163), (62, 10715), (62, 4061), (55, 2014), (46, 10380), (53, 5254), (51, 10513), (61, 9740), (58, 9817), (63, 13914), (52, 14130), (61, 13532), (49, 2886), (43, 6872), (53, 11279), (53, 8170), (60, 12831), (59, 11393), (61, 14926), (47, 536), (48, 16341), (52, 1545), (56, 10683), (62, 445), (53, 735), (54, 11360), (60, 1567), (59, 7811), (52, 3604), (61, 2057), (60, 6015), (63, 6277), (63, 10542), (62, 7300), (53, 11135), (50, 89), (48, 2160), (63, 8693), (49, 6186), (48, 4448), (49, 1274), (51, 90), (56, 11149), (58, 14505), (58, 12290), (55, 2583), (63, 6496), (52, 16484), (50, 3417), (51, 8986), (51, 12705), (51, 7436), (60, 10383), (59, 8106), (43, 1498), (51, 2460), (46, 1909), (61, 5058), (51, 2542), (61, 2663), (50, 4841), (43, 634), (47, 2398), (61, 2754), (47, 6126), (60, 4497), (52, 8093), (62, 2673), (57, 2622), (36, 11587), (50, 6610), (56, 6385), (60, 1034), (53, 4531), (51, 5043), (50, 14112), (54, 8676), (50, 6192), (61, 6802), (48, 5700), (61, 7705), (54, 8987), (50, 17129), (52, 10765), (63, 4560), (50, 9819), (61, 2018), (55, 6178), (61, 6411), (52, 7302), (18, 8609), (52, 9709), (49, 2611), (54, 8495), (62, 2440), (55, 16169), (50, 9913), (51, 2357), (53, 11443), (60, 16824), (50, 9266), (53, 11150), (52, 5727), (55, 4850), (49, 3733), (51, 14853)


]
import torch
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm

from dance.core.kn_code import load_model


save_dir = PROJECT_RESULTS_DIR
os.makedirs(save_dir, exist_ok=True)
PROMPT_DIR = PROJECT_PROMPT_DIR
DATA_DIR = PROJECT_DATA_DIR


def get_collaboration_device_map():
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


def compute_neuron_contributions(model_name, condition, specified_neurons, num_prob):
    print(f"--- Initializing Model: {model_name} ---")
    kn = load_model(model_name, device_map=get_collaboration_device_map())

    hf_device_map = getattr(kn.model, "hf_device_map", None)
    if hf_device_map:
        dispatched_devices = sorted(
            {str(device) for device in hf_device_map.values()},
            key=str,
        )
        print(
            "Accelerate model dispatch devices: "
            f"{dispatched_devices} ({len(hf_device_map)} mapped modules)"
        )


    model_name_clean = model_name
    if '/' in model_name:
        model_name_clean = os.path.basename(model_name)

    if "llama" in model_name_clean.lower():
        model_name_clean = model_name_clean.replace("llama", "").strip("-_")
    elif "Qwen" in model_name_clean:
        model_name_clean = model_name_clean.replace("Qwen", "").strip("-_")
    elif "mistral" in model_name_clean.lower():
        model_name_clean = model_name_clean.replace("mistral", "").strip("-_")

    if not model_name_clean:
        model_name_clean = "custom_model"


    print(f"--- Loading Data for Condition: {condition} ---")

    try:
        _df = None

        if args.add == '':
            file_path = os.path.join(DATA_DIR, 'bias_importance_data.csv')
            if os.path.exists(file_path):
                _df = pd.read_csv(file_path)
                print(f"[OK] Loaded specified dataset: {file_path}")
            else:
                raise FileNotFoundError(f"Specified dataset not found: {file_path}")
        else:
            raise ValueError(f"Unknown --add argument value: '{args.add}'")

    except Exception as e:
        print(f"\n[CRITICAL] Data Loading Failed: {e}")
        raise e

    if _df is None or len(_df) == 0:
        raise ValueError("Dataframe is empty or None. Cannot proceed with analysis.")

    print(f"Successfully loaded dataframe with {len(_df)} rows.")


    start, end = args.start, args.end
    if end == -1 or end > len(_df):
        end = len(_df)

    if start > 0 or end < len(_df):
        print(f"Slicing data from index {start} to {end}...")
        _df = _df.iloc[start:end]
        print(f"Sliced dataframe size: {len(_df)}")

    neurons_to_analyze = specified_neurons

    print(f"\nNumber of neurons selected for analysis: {len(neurons_to_analyze)}")
    print(f"Neuron list: {neurons_to_analyze[:10]}{'...' if len(neurons_to_analyze) > 10 else ''}")


    n_layers = kn.n_layers()
    layer_intermediate_sizes = {}
    for layer_idx, neuron_idx in neurons_to_analyze:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} is outside the valid range [0, {n_layers-1}]")
        if layer_idx not in layer_intermediate_sizes:
            layer_intermediate_sizes[layer_idx] = kn.intermediate_size(layer_idx)
        intermediate_size = layer_intermediate_sizes[layer_idx]
        if neuron_idx < 0 or neuron_idx >= intermediate_size:
            raise ValueError(
                f"Layer {layer_idx} neuron index {neuron_idx} is outside the valid range "
                f"[0, {intermediate_size-1}]"
            )

    print(
        f"Model layers: {n_layers}, MLP intermediate dimensions: "
        f"{sorted(set(layer_intermediate_sizes.values()))}"
    )


    neuron_contributions = {neuron: [] for neuron in neurons_to_analyze}
    is_qwen35 = 'qwen3.5' in model_name.lower() or 'qwen3_5' in model_name.lower()
    neurons_by_layer = {}
    for layer_idx, neuron_idx in neurons_to_analyze:
        neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)
    if is_qwen35:
        print(
            f"[Qwen3.5 optimization] Computing only the layers containing selected neurons: "
            f"{len(neurons_by_layer)}/{n_layers} layers; skipping frequent CUDA-cache cleanup inside the IG loop"
        )


    print("--- Starting Knowledge Neuron Analysis for Specified Neurons ---")
    for idx, row in tqdm(_df.iterrows(), total=len(_df), desc="Processing Samples"):

        if not is_qwen35 and idx % 5 == 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        question = row['question']
        target = row['gold']


        prompt_file_main = f'{PROMPT_DIR}/evaluate_know_new.txt'
        prompt_file_fallback = f'{PROMPT_DIR}/evaluate_s.txt'

        if os.path.exists(prompt_file_main):
            with open(prompt_file_main, 'r', encoding='utf-8') as f:
                instruction = f.read()
        elif os.path.exists(prompt_file_fallback):
            with open(prompt_file_fallback, 'r', encoding='utf-8') as f:
                instruction = f.read()
        else:
            raise FileNotFoundError(f"Prompt file not found: neither {prompt_file_main} nor {prompt_file_fallback}")

        q_add = f"{question}"
        source_prompt = f"""{instruction}
Question: {q_add}"""

        if is_qwen35:


            for layer_idx, neuron_indices in neurons_by_layer.items():
                layer_scores = kn.get_scores_for_layer(
                    source_prompt,
                    target,
                    layer_idx=layer_idx,
                    batch_size=args.qwen_batch_size,
                    steps=args.qwen_steps,
                    attribution_method="integrated_grads",
                    aggressive_cleanup=False,
                    validate_finite=False,
                ).detach().cpu()
                for neuron_idx in neuron_indices:
                    neuron_contributions[(layer_idx, neuron_idx)].append(
                        layer_scores[neuron_idx].item()
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:

            delta = kn.search_kns_modi(
                source_prompt,
                target,
                n_prompts=num_prob,
                batch_size=4,
                verbose=args.verbose,
            )
            att_score = delta.detach().cpu()
            for neuron in neurons_to_analyze:
                layer_idx, neuron_idx = neuron
                score = att_score[layer_idx, neuron_idx].item()
                neuron_contributions[neuron].append(score)


    print("\n" + "="*60)
    print("Selected-neuron contribution statistics")
    print("="*60)


    n_samples = len(_df)
    if end != -1 and end < len(_df):
        n_samples = end - start


    results_data = []
    for neuron in neurons_to_analyze:
        layer_idx, neuron_idx = neuron
        contributions = neuron_contributions[neuron]

        if len(contributions) > 0:
            mean_contrib = np.mean(contributions)


            row = {
                'layer': layer_idx,
                'neuron': neuron_idx,
                'mean_contribution': mean_contrib,
                'num_samples': len(contributions)
            }

            for sample_idx, score in enumerate(contributions):
                row[f'sample_{sample_idx}'] = score

            results_data.append(row)

            print(f"\nNeuron (Layer {layer_idx}, Neuron {neuron_idx}):")
            print(f"  Samples: {len(contributions)}")
            print(f"  Mean contribution: {mean_contrib:.6f}")
        else:
            row = {
                'layer': layer_idx,
                'neuron': neuron_idx,
                'mean_contribution': 0.0,
                'num_samples': 0
            }
            for sample_idx in range(n_samples):
                row[f'sample_{sample_idx}'] = 0.0
            results_data.append(row)


    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('mean_contribution', ascending=False).reset_index(drop=True)




    total_neurons = len(results_df)
    results_df['rank'] = range(1, total_neurons + 1)
    results_df['weight'] = (total_neurons - results_df['rank'] + 1) / total_neurons

    print("\n" + "="*60)
    print("Neuron weights ranked by importance")
    print("="*60)
    print(results_df[['layer', 'neuron', 'mean_contribution', 'rank', 'weight']].head(10).to_string(index=False))
    print(f"...\nTotal: {total_neurons} neurons")


    save_filename = (
        f'specified_neurons_contributions_'
        f'{len(neurons_to_analyze)}neurons_'
        f'{model_name_clean}_'
        f'{args.add}.csv'
    )
    save_path = os.path.join(save_dir, save_filename)



    sample_cols = [col for col in results_df.columns if col.startswith('sample_')]
    ordered_cols = ['layer', 'neuron', 'mean_contribution', 'num_samples',
                    'rank', 'weight'] + sample_cols
    results_df[ordered_cols].to_csv(save_path, index=False)
    print(f"\nFull results (including per-example attributions) saved to: {save_path}")

    print("\nDone.")

    return results_data


def main():
    parser = argparse.ArgumentParser(description='Contribution analysis for selected neurons')
    parser.add_argument('--condition', type=str, default='Gender_identity')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--num_probs', '-n', type=int, default=5)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--add', type=str, default='')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument(
        '--qwen_steps', type=int, default=20,
        help='Qwen3.5 integrated-gradient steps (default: 20 to preserve attribution precision)'
    )
    parser.add_argument(
        '--qwen_batch_size', type=int, default=4,
        help='Qwen3.5 integrated-gradient microbatch size; must divide qwen_steps'
    )

    global args
    args = parser.parse_args()

    if args.qwen_steps <= 0 or args.qwen_batch_size <= 0:
        parser.error('--qwen_steps and --qwen_batch_size must be positive integers')
    if args.qwen_steps % args.qwen_batch_size != 0:
        parser.error('--qwen_steps must be divisible by --qwen_batch_size')


    specified_neurons = SPECIFIED_NEURONS

    if not specified_neurons:
        raise ValueError("Define neurons to analyze in SPECIFIED_NEURONS at the top of this script")

    print(f"Using the predefined neuron list: {len(specified_neurons)} neurons")

    compute_neuron_contributions(
        args.model_name,
        args.condition,
        specified_neurons,
        args.num_probs
    )


if __name__ == '__main__':
    main()


