
SPECIFIED_NEURONS = [






































    (46, 7410), (49, 8062), (55, 17163), (49, 14899), (54, 6966), (57, 4649), (50, 11755), (60, 5221), (51, 10670), (51, 2640), (55, 5963), (52, 9921), (59, 8551), (54, 11926), (49, 7808), (57, 75), (52, 5247), (52, 5073), (44, 14484), (50, 12340), (61, 15458), (61, 13844), (51, 8320), (58, 13366), (62, 4821), (59, 6656), (32, 8316), (58, 8881), (61, 7547), (52, 13960), (59, 7330), (44, 9805), (49, 9461), (51, 6370), (55, 11908), (47, 14064), (63, 2870), (52, 11418), (50, 804), (56, 14085), (46, 5291), (48, 6847), (61, 12214), (60, 3877), (56, 2519), (63, 11958), (60, 1909), (47, 1323), (50, 7375), (61, 13910), (49, 2434), (61, 12155), (50, 6902), (55, 16157), (51, 7257), (53, 14064), (57, 8146), (52, 4748), (51, 15906), (62, 396), (55, 12157), (49, 15415), (54, 17025), (43, 11655), (50, 2191), (56, 5456), (53, 13200), (61, 5809), (62, 13813), (55, 728), (58, 8220), (62, 16876), (51, 4013), (49, 7769), (49, 9209), (44, 3770), (48, 15992), (56, 2121), (48, 4298), (60, 489), (55, 12742), (50, 16392), (50, 5711), (50, 1336), (49, 3049), (61, 15855), (48, 11153), (45, 16295), (51, 15981), (62, 535), (51, 4818), (51, 8125), (51, 5190), (49, 843), (59, 15037), (42, 16525), (61, 8127), (58, 2990), (56, 11630), (58, 14693), (63, 17405), (61, 12767), (49, 7707), (41, 13429), (47, 6724), (57, 7412), (58, 7804), (62, 9925), (61, 1265), (60, 1623), (54, 12508), (49, 13131), (49, 12356), (57, 3064), (54, 7669), (54, 7788), (49, 5165), (49, 5467), (51, 8374), (45, 13081), (62, 14775), (56, 5344), (51, 11391), (49, 1293), (56, 1088), (50, 6124), (55, 3978), (43, 11625), (59, 2655), (51, 9423), (54, 11854), (46, 897), (61, 3436), (34, 17129), (49, 8267), (61, 7893), (50, 16780), (51, 9816), (47, 11085), (61, 348), (56, 13376), (49, 11659), (49, 12150), (62, 3922), (55, 8529), (46, 6095), (49, 4294), (59, 369), (48, 13173), (48, 8316), (49, 3092), (52, 4320), (47, 6801), (55, 15201), (62, 7778), (42, 545), (55, 6204), (55, 10881), (54, 13461), (59, 10278), (53, 1478), (53, 14557), (49, 11997), (46, 13624), (62, 12964), (59, 10186), (55, 16906), (37, 5198), (54, 10382), (25, 9809), (50, 12773), (50, 17267), (49, 13821), (49, 15380), (53, 5471), (44, 10551), (63, 11068), (36, 16328), (51, 2575), (58, 9433), (60, 11163), (62, 10715), (62, 4061), (55, 2014), (46, 10380), (53, 5254), (51, 10513), (61, 9740), (58, 9817), (63, 13914), (52, 14130), (61, 13532), (49, 2886), (43, 6872), (53, 11279), (53, 8170), (60, 12831), (59, 11393), (61, 14926), (47, 536), (48, 16341), (52, 1545), (56, 10683), (62, 445), (53, 735), (54, 11360), (60, 1567), (59, 7811), (52, 3604), (61, 2057), (60, 6015), (63, 6277), (63, 10542), (62, 7300), (53, 11135), (50, 89), (48, 2160), (63, 8693), (49, 6186), (48, 4448), (49, 1274), (51, 90), (56, 11149), (58, 14505), (58, 12290), (55, 2583), (63, 6496), (52, 16484), (50, 3417), (51, 8986), (51, 12705), (51, 7436), (60, 10383), (59, 8106), (43, 1498), (51, 2460), (46, 1909), (61, 5058), (51, 2542), (61, 2663), (50, 4841), (43, 634), (47, 2398), (61, 2754), (47, 6126), (60, 4497), (52, 8093), (62, 2673), (57, 2622), (36, 11587), (50, 6610), (56, 6385), (60, 1034), (53, 4531), (51, 5043), (50, 14112), (54, 8676), (50, 6192), (61, 6802), (48, 5700), (61, 7705), (54, 8987), (50, 17129), (52, 10765), (63, 4560), (50, 9819), (61, 2018), (55, 6178), (61, 6411), (52, 7302), (18, 8609), (52, 9709), (49, 2611), (54, 8495), (62, 2440), (55, 16169), (50, 9913), (51, 2357), (53, 11443), (60, 16824), (50, 9266), (53, 11150), (52, 5727), (55, 4850), (49, 3733), (51, 14853)


]


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import torch
import sys
import collections


from llm_utils import LLM_KN
from prettytable import PrettyTable


DATA_DIR = 'data'
SAVE_DIR = "results/"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_general_device_map():
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


def get_model_type(model_name):
    model_name_lower = model_name.lower()
    if 'llama' in model_name_lower:
        return 'llama'
    elif 'qwen' in model_name_lower:
        return 'Qwen'
    elif 'gemma' in model_name_lower:
        return 'gemma'
    elif 'mistral' in model_name_lower:
        return 'mistral'
    else:
        return 'bert'


class NeuronSuppressor:

    def __init__(self, model, layer_idx, neuron_idx, model_type='llama'):
        self.model = model
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.model_type = model_type
        self.hook_handle = None

    def register_hook(self):
        if self.model_type == 'llama':


            mlp = self.model.model.layers[self.layer_idx].mlp
        elif self.model_type == 'Qwen':
            mlp = self.model.model.layers[self.layer_idx].mlp
        elif self.model_type == 'bert':
            mlp = self.model.bert.encoder.layer[self.layer_idx].intermediate


        def hook_fn(module, input, output):


            if len(output.shape) == 3:
                output[:, :, self.neuron_idx] = 0.0
            return output


        if self.model_type == 'llama':

            self.hook_handle = mlp.up_proj.register_forward_hook(hook_fn)
        elif self.model_type == 'Qwen':

            self.hook_handle = mlp.up_proj.register_forward_hook(hook_fn)
        elif self.model_type == 'bert':

            self.hook_handle = mlp.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def evaluate_obqa(perspectiveModel, all_rows, num_probs, show_progress=True):
    correct_count = 0


    rows_to_eval = []
    for idx, row in all_rows.iterrows():
        if idx >= num_probs:
            break
        rows_to_eval.append((idx, row))


    if show_progress:
        iterator = tqdm(rows_to_eval, desc="Evaluating samples", leave=False)
    else:
        iterator = rows_to_eval

    for idx, row in iterator:
        question_stem = str(row['question_stem'])
        choices_dict = row['choices']
        answer_key = str(row['answerKey'])

        choice_texts = choices_dict['text']
        choice_labels = choices_dict['label']

        instruction = "Based on the question, select the best answer from the options provided."
        choices_str = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])
        prompt = f"{instruction}\n\nQuestion: {question_stem}\n\nOptions:\n{choices_str}"



        option_probs = perspectiveModel.getProb_batch(
            prompt,
            choice_texts,
            correct_causal_alignment=True,
        )
        probs = {
            label: option_probs[text]
            for label, text in zip(choice_labels, choice_texts)
        }

        model_selected = max(probs, key=probs.get)
        if model_selected == answer_key:
            correct_count += 1

    return correct_count


def run_baseline(model_name, num_probs=None, temperature=0.2, eight_bit=False, verbose=False):
    print(f"\n{'='*60}")
    print(f"Running Baseline OBQA Evaluation")
    print(f"{'='*60}")

    parquet_path = os.path.join(DATA_DIR, 'OBQA11.parquet')
    all_rows = pd.read_parquet(parquet_path)

    if num_probs is None:
        num_probs = len(all_rows)
    num_probs = min(num_probs, len(all_rows))

    print(f"Loading model: {model_name}")
    perspectiveModel = LLM_KN(
        model_name,
        temperature=temperature,
        load_in_8bit=eight_bit,
        verbose=verbose,
        device_map=get_general_device_map(),
    )

    print(f"Running baseline evaluation...")
    correct_count = evaluate_obqa(perspectiveModel, all_rows, num_probs, show_progress=True)

    baseline_accuracy = (correct_count / num_probs * 100) if num_probs > 0 else 0
    print(f"\nBaseline Accuracy: {baseline_accuracy:.2f}%")

    return baseline_accuracy, perspectiveModel


def run_neuron_ablation(model_name, specified_neurons, num_probs=None, temperature=0.2, eight_bit=False, verbose=False):
    print(f"\n{'='*60}")
    print(f"Running Neuron Ablation Experiment")
    print(f"Number of selected neurons: {len(specified_neurons)}")
    print(f"{'='*60}")

    parquet_path = os.path.join(DATA_DIR, 'OBQA11.parquet')
    all_rows = pd.read_parquet(parquet_path)

    if num_probs is None:
        num_probs = len(all_rows)
    num_probs = min(num_probs, len(all_rows))


    baseline_accuracy, perspectiveModel = run_baseline(model_name, num_probs, temperature, eight_bit, verbose)

    model = perspectiveModel.kn.model
    model_type = get_model_type(model_name)

    results = []

    print(f"\nStarting the single-neuron ablation experiments...")

    for i, (layer_idx, neuron_idx) in enumerate(tqdm(specified_neurons, desc="Ablating neurons")):

        suppressor = NeuronSuppressor(model, layer_idx, neuron_idx, model_type)
        suppressor.register_hook()


        correct_count = evaluate_obqa(perspectiveModel, all_rows, num_probs, show_progress=False)


        suppressor.remove_hook()

        accuracy = (correct_count / num_probs * 100) if num_probs > 0 else 0
        impact = baseline_accuracy - accuracy

        results.append({
            'layer': layer_idx,
            'neuron': neuron_idx,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'baseline_accuracy': baseline_accuracy,
            'impact': impact
        })

        if verbose and i < 5:
            print(f"  Neuron ({layer_idx}, {neuron_idx}): Accuracy={accuracy:.2f}%, Impact={impact:.2f}%")


    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('impact', ascending=False).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Ablation results summary (ranked by impact)")
    print(f"{'='*60}")
    print(results_df.head(20).to_string(index=False))

    return results_df, baseline_accuracy


def main():
    parser = argparse.ArgumentParser(description='OBQA neuron-ablation experiment')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--num_probs', '-n', type=int, default=None, help='Number of evaluation examples (default: all examples)')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature')
    parser.add_argument('--eight_bit', action='store_true', help='Load the model in 8-bit precision')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    specified_neurons = SPECIFIED_NEURONS

    if not specified_neurons:
        raise ValueError("Define neurons to test in SPECIFIED_NEURONS at the top of this script")

    print(f"Using the predefined neuron list: {len(specified_neurons)} neurons")

    results_df, baseline_accuracy = run_neuron_ablation(
        args.model_name,
        specified_neurons,
        args.num_probs,
        args.temperature,
        args.eight_bit,
        args.verbose
    )

    model_short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    save_filename = f'general_importance_results_{model_short}.csv'
    save_path = os.path.join(SAVE_DIR, save_filename)
    results_df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")

    print(f"\nBaseline accuracy: {baseline_accuracy:.2f}%")
    print(f"Neuron with the largest impact: Layer {results_df.iloc[0]['layer']}, Neuron {results_df.iloc[0]['neuron']}, Impact: {results_df.iloc[0]['impact']:.2f}%")


if __name__ == '__main__':
    main()



