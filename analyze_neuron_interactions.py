
import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm

from kn_code import load_model
from kn_code.patch import patch_ff_layer, unpatch_ff_layers


save_dir = "results/"
os.makedirs(save_dir, exist_ok=True)
PROMPT_DIR = 'prompt_instructions'
DATA_DIR = 'data'


def get_collaboration_device_map(model_name):
    normalized_name = (
        model_name.lower().replace("_", "").replace("-", "").replace(".", "")
    )
    is_qwen35_27b = "qwen35" in normalized_name and "27b" in normalized_name
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if is_qwen35_27b and visible_gpus >= 2:
        print(
            f"Detected Qwen3.5-27B and {visible_gpus} visible GPUs; "
            "using device_map='balanced' for model parallelism."
        )
        return "balanced"

    if is_qwen35_27b:
        print(
            f"Detected Qwen3.5-27B but only {visible_gpus} visible GPU(s); "
            "using the original single-device loading."
        )
    return None


def compute_prob_with_masked_neurons(model, kn, prompt, target, masked_neurons, use_enhanced=False, enhance_value=3.0):
    encoded_input, mask_idx, target_label = kn._prepare_inputs(prompt, target)


    layers_to_patch = list(set([n[0] for n in masked_neurons]))


    for layer_idx in layers_to_patch:
        layer_neurons = [n[1] for n in masked_neurons if n[0] == layer_idx]

        if use_enhanced:

            patch_ff_layer(
                model,
                mask_idx=mask_idx,
                layer_idx=layer_idx,
                mode="enhance",
                neurons=[(layer_idx, n) for n in layer_neurons],
                _value=enhance_value,
                transformer_layers_attr=kn.transformer_layers_attr,
                ff_attrs=kn.input_ff_attr,
            )
        else:

            patch_ff_layer(
                model,
                mask_idx=mask_idx,
                layer_idx=layer_idx,
                mode="suppress",
                neurons=[(layer_idx, n) for n in layer_neurons],
                transformer_layers_attr=kn.transformer_layers_attr,
                ff_attrs=kn.input_ff_attr,
            )


    with torch.no_grad():
        outputs = model(**encoded_input)
        probs = torch.softmax(outputs.logits[:, mask_idx, :], dim=-1, dtype=torch.float32)
        target_idx = target_label if isinstance(target_label, int) else target_label[0]
        target_prob = probs[0, target_idx].item()


    if layers_to_patch:
        unpatch_ff_layers(
            model,
            layer_indices=layers_to_patch,
            transformer_layers_attr=kn.transformer_layers_attr,
            ff_attrs=kn.input_ff_attr,
        )

    return target_prob


def compute_pairwise_synergy_simple(model, kn, source_prompt, target, neuron_A, neuron_B, enhance_value=2.0):
    layer_A, neuron_A_idx = neuron_A
    layer_B, neuron_B_idx = neuron_B

    def probability_with_enhancement(neurons):
        neurons_by_layer = {}
        for layer_idx, neuron_idx in neurons:
            neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)

        patched_layers = []
        try:
            for layer_idx, neuron_indices in neurons_by_layer.items():


                unique_indices = list(dict.fromkeys(neuron_indices))
                patch_ff_layer(
                    model,
                    mask_idx=mask_idx,
                    layer_idx=layer_idx,
                    mode="enhance",
                    neurons=[(layer_idx, idx) for idx in unique_indices],
                    _value=enhance_value,
                    transformer_layers_attr=kn.transformer_layers_attr,
                    ff_attrs=kn.input_ff_attr,
                )
                patched_layers.append(layer_idx)

            with torch.no_grad():
                enhanced_outputs = model(**encoded_input)
                enhanced_probs = torch.softmax(
                    enhanced_outputs.logits[:, mask_idx, :],
                    dim=-1,
                    dtype=torch.float32,
                )
                return enhanced_probs[0, target_idx].item()
        finally:


            if patched_layers:
                unpatch_ff_layers(
                    model,
                    layer_indices=list(dict.fromkeys(patched_layers)),
                    transformer_layers_attr=kn.transformer_layers_attr,
                    ff_attrs=kn.input_ff_attr,
                )


    encoded_input, mask_idx, target_label = kn._prepare_inputs(source_prompt, target)
    with torch.no_grad():
        outputs = model(**encoded_input)
        probs = torch.softmax(outputs.logits[:, mask_idx, :], dim=-1, dtype=torch.float32)
        target_idx = target_label if isinstance(target_label, int) else target_label[0]
        prob_baseline = probs[0, target_idx].item()


    prob_A_enhanced = probability_with_enhancement([neuron_A])
    prob_B_enhanced = probability_with_enhancement([neuron_B])
    prob_AB_enhanced = probability_with_enhancement([neuron_A, neuron_B])


    effect_A = prob_A_enhanced - prob_baseline
    effect_B = prob_B_enhanced - prob_baseline
    effect_AB = prob_AB_enhanced - prob_baseline
    expected_joint = effect_A + effect_B
    synergy = effect_AB - expected_joint

    return {
        'layer_A': layer_A,
        'neuron_A': neuron_A_idx,
        'layer_B': layer_B,
        'neuron_B': neuron_B_idx,
        'prob_baseline': prob_baseline,
        'prob_A_only': prob_A_enhanced,
        'prob_B_only': prob_B_enhanced,
        'prob_AB_enhanced': prob_AB_enhanced,
        'effect_A': effect_A,
        'effect_B': effect_B,
        'effect_AB': effect_AB,
        'expected_joint': expected_joint,
        'synergy': synergy,
        'enhance_value': enhance_value
    }


def analyze_neuron_pairs(model_name, neuron_pairs_df, num_samples=1, enhance_value=2.0):
    print(f"--- Initializing Model: {model_name} ---")
    print(f"--- Enhance Value: {enhance_value} ---")
    kn = load_model(
        model_name,
        device_map=get_collaboration_device_map(model_name),
    )

    hf_device_map = getattr(kn.model, "hf_device_map", None)
    if hf_device_map:
        dispatched_devices = sorted({str(device) for device in hf_device_map.values()})
        print(
            f"Accelerate model dispatch devices: {dispatched_devices} "
            f"({len(hf_device_map)} mapped modules)"
        )




    n_layers = kn.n_layers()
    layer_widths = {layer_idx: kn.intermediate_size(layer_idx) for layer_idx in range(n_layers)}
    invalid_coordinates = []
    for row_idx, row in neuron_pairs_df.iterrows():
        for side in ('i', 'j'):
            layer_idx = int(row[f'layer_{side}'])
            neuron_idx = int(row[f'neuron_{side}'])
            if layer_idx < 0 or layer_idx >= n_layers:
                invalid_coordinates.append(
                    (int(row_idx), side, layer_idx, neuron_idx, f'layer must be in [0, {n_layers - 1}]')
                )
            elif neuron_idx < 0 or neuron_idx >= layer_widths[layer_idx]:
                invalid_coordinates.append(
                    (
                        int(row_idx), side, layer_idx, neuron_idx,
                        f'neuron must be in [0, {layer_widths[layer_idx] - 1}]',
                    )
                )

    if invalid_coordinates:
        examples = '\n'.join(
            f'  row={row_idx}, side={side}, coordinate=({layer_idx}, {neuron_idx}): {reason}'
            for row_idx, side, layer_idx, neuron_idx, reason in invalid_coordinates[:10]
        )
        raise ValueError(
            f'Candidate neuron file contains {len(invalid_coordinates)} coordinates '
            f'that are invalid for {model_name}. Model layers: {n_layers}.\n'
            f'First invalid coordinates:\n{examples}\n'
            'Please use the neuron-pair CSV extracted from this exact model.'
        )

    print(
        f'--- Validated neuron coordinates: {n_layers} layers, '
        f'intermediate widths={sorted(set(layer_widths.values()))} ---'
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


    prompt_file_main = f'{PROMPT_DIR}/evaluate_know_new.txt'
    prompt_file_fallback = f'{PROMPT_DIR}/evaluate_s.txt'

    if os.path.exists(prompt_file_main):
        with open(prompt_file_main, 'r', encoding='utf-8') as f:
            instruction = f.read()
    elif os.path.exists(prompt_file_fallback):
        with open(prompt_file_fallback, 'r', encoding='utf-8') as f:
            instruction = f.read()
    else:
        raise FileNotFoundError(f"Prompt file not found")


    data_path = os.path.join(DATA_DIR, 'know_bias_data_extend.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df_data = pd.read_csv(data_path)
    print(f"--- Loaded {len(df_data)} samples from data ---")
    print(f"--- Analyzing {len(neuron_pairs_df)} neuron pairs ---")


    all_pair_results = []


    for pair_idx, row in tqdm(neuron_pairs_df.iterrows(), total=len(neuron_pairs_df), desc="Analyzing Neuron Pairs"):
        neuron_A = (int(row['layer_i']), int(row['neuron_i']))
        neuron_B = (int(row['layer_j']), int(row['neuron_j']))

        pair_sample_results = []


        for sample_idx in range(min(num_samples, len(df_data))):
            question = df_data.iloc[sample_idx]['question']
            target = df_data.iloc[sample_idx]['gold']

            source_prompt = f"""{instruction}
Question: {question}"""

            try:
                result = compute_pairwise_synergy_simple(
                    kn.model,
                    kn,
                    source_prompt,
                    target,
                    neuron_A,
                    neuron_B,
                    enhance_value=enhance_value
                )
                pair_sample_results.append(result)

            except Exception as e:
                print(
                    f"\nError processing pair {neuron_A} - {neuron_B}, "
                    f"sample {sample_idx}: {type(e).__name__}: {e!r}"
                )
                pair_sample_results.append({
                    'layer_A': neuron_A[0],
                    'neuron_A': neuron_A[1],
                    'layer_B': neuron_B[0],
                    'neuron_B': neuron_B[1],
                    'prob_baseline': 0.0,
                    'prob_A_only': 0.0,
                    'prob_B_only': 0.0,
                    'prob_AB_enhanced': 0.0,
                    'effect_A': 0.0,
                    'effect_B': 0.0,
                    'effect_AB': 0.0,
                    'expected_joint': 0.0,
                    'synergy': 0.0
                })


            if sample_idx % 5 == 0:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()



        avg_synergy = sum(r['synergy'] for r in pair_sample_results) / len(pair_sample_results)
        avg_prob_baseline = sum(r['prob_baseline'] for r in pair_sample_results) / len(pair_sample_results)
        avg_prob_A = sum(r['prob_A_only'] for r in pair_sample_results) / len(pair_sample_results)
        avg_prob_B = sum(r['prob_B_only'] for r in pair_sample_results) / len(pair_sample_results)
        avg_prob_AB = sum(r['prob_AB_enhanced'] for r in pair_sample_results) / len(pair_sample_results)
        avg_effect_A = sum(r['effect_A'] for r in pair_sample_results) / len(pair_sample_results)
        avg_effect_B = sum(r['effect_B'] for r in pair_sample_results) / len(pair_sample_results)
        avg_effect_AB = sum(r['effect_AB'] for r in pair_sample_results) / len(pair_sample_results)
        avg_expected = sum(r['expected_joint'] for r in pair_sample_results) / len(pair_sample_results)

        pair_result = {
            'layer_A': neuron_A[0],
            'neuron_A': neuron_A[1],
            'layer_B': neuron_B[0],
            'neuron_B': neuron_B[1],
            'prob_baseline': avg_prob_baseline,
            'prob_A_only': avg_prob_A,
            'prob_B_only': avg_prob_B,
            'prob_AB_enhanced': avg_prob_AB,
            'effect_A': avg_effect_A,
            'effect_B': avg_effect_B,
            'effect_AB': avg_effect_AB,
            'expected_joint': avg_expected,
            'synergy': avg_synergy,
            'num_samples': len(pair_sample_results)
        }


        if avg_synergy > 0:
            pair_result['relationship'] = 'synergistic'
        elif avg_synergy < 0:
            pair_result['relationship'] = 'suppressive'
        else:
            pair_result['relationship'] = 'independent'

        all_pair_results.append(pair_result)


        if pair_idx % 10 == 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    results_df = pd.DataFrame(all_pair_results)


    synergistic_count = (results_df['relationship'] == 'synergistic').sum()
    suppressive_count = (results_df['relationship'] == 'suppressive').sum()


    print("\n" + "="*60)
    print("Synergy 值分布统计")
    print("="*60)
    print(f"  Min:  {results_df['synergy'].min():.6f}")
    print(f"  Max:  {results_df['synergy'].max():.6f}")
    print(f"  Mean: {results_df['synergy'].mean():.6f}")
    print(f"  Std:  {results_df['synergy'].std():.6f}")
    print(f"  Median: {results_df['synergy'].median():.6f}")

    print("\n" + "="*60)
    print("成对神经元协同/抑制分析结果统计")
    print("="*60)
    print(f"总神经元对数: {len(results_df)}")
    print(f"协同 (Synergistic): {synergistic_count} ({synergistic_count/len(results_df)*100:.1f}%)")
    print(f"抑制 (Suppressive): {suppressive_count} ({suppressive_count/len(results_df)*100:.1f}%)")


    save_filename = f'pairwise_neuron_synergy_{model_name_clean}_enhance_{enhance_value}.csv'
    save_path = os.path.join(save_dir, save_filename)
    results_df.to_csv(save_path, index=False)
    print(f"\n结果已保存到: {save_path}")


    if synergistic_count > 0:
        print("\n" + "="*60)
        print("协同效应最强的神经元对 (Top 10)")
        print("="*60)
        top_synergistic = results_df[results_df['relationship'] == 'synergistic'].nlargest(10, 'synergy')
        print(top_synergistic[['layer_A', 'neuron_A', 'layer_B', 'neuron_B', 'synergy', 'relationship']].to_string(index=False))

    if suppressive_count > 0:
        print("\n" + "="*60)
        print("抑制效应最强的神经元对 (Top 10)")
        print("="*60)
        top_suppressive = results_df[results_df['relationship'] == 'suppressive'].nsmallest(10, 'synergy')
        print(top_suppressive[['layer_A', 'neuron_A', 'layer_B', 'neuron_B', 'synergy', 'relationship']].to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser(description='成对神经元协同/抑制分析')
    parser.add_argument('--condition', type=str, default='Gender_identity')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--num_samples', '-n', type=int, default=1, help='每个神经元对分析的样本数量')
    parser.add_argument('--enhance_value', '-e', type=float, default=2.0, help='神经元增强倍数')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--add', type=str, default='')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--pairs_file', type=str, default='data/candidate_neuron_pairs.csv')

    global args
    args = parser.parse_args()


    pairs_file = args.pairs_file
    if not os.path.isabs(pairs_file):
        pairs_file = os.path.join(os.path.dirname(__file__), pairs_file)

    print(f"--- Loading Neuron Pairs from: {pairs_file} ---")
    neuron_pairs_df = pd.read_csv(pairs_file)


    start, end = args.start, args.end
    if end == -1 or end > len(neuron_pairs_df):
        end = len(neuron_pairs_df)

    if start > 0 or end < len(neuron_pairs_df):
        print(f"Slicing pairs from index {start} to {end}...")
        neuron_pairs_df = neuron_pairs_df.iloc[start:end]
        print(f"Sliced pairs size: {len(neuron_pairs_df)}")

    print(f"Loaded {len(neuron_pairs_df)} neuron pairs")


    results_df = analyze_neuron_pairs(
        args.model_name,
        neuron_pairs_df,
        args.num_samples,
        args.enhance_value
    )

    print("\nDone.")


if __name__ == '__main__':
    main()


