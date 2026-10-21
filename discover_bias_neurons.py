import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import ttest_ind
from datasets import load_dataset

from kn_code.patch import unpatch_ff_layers
from kn_code import load_model
from kn_code.knowledge_neurons import get_top_k_neurons
from time import strftime
import argparse
import ast
import os
import pandas as pd
from tqdm import tqdm


save_dir = "results/"
os.makedirs(save_dir, exist_ok=True)
PROMPT_DIR = 'prompt_instructions'
DATA_DIR = 'data'


def get_run_device_map():
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpus >= 2:
        print(
            f"Detected {visible_gpus} visible GPUs; "
            "using device_map='balanced' for multi-GPU model parallelism."
        )
        return "balanced"

    print(f"Detected {visible_gpus} visible GPU(s); using the original single-device loading.")
    return None


def know_bias_analysis_temp_all_new(model_name, condition, num_prob):
    print(f"--- Initializing Model: {model_name} ---")
    kn = load_model(model_name, device_map=get_run_device_map())
    is_qwen35 = getattr(kn.model.config, 'model_type', '').lower() in {
        'qwen3_5', 'qwen3_5_text'
    }
    if is_qwen35:
        print(
            "Qwen3.5 attribution optimization enabled: "
            f"steps={args.steps}, batch_size={args.qwen_batch_size}, "
            "diagnostic cache flushes/full-logit scans disabled."
        )

    _df = None
    th_val = args.threshold
    p_val = args.p_value


    model_name_clean = model_name
    if '/' in model_name:

        model_name_clean = os.path.basename(model_name)


    if "llama" in model_name_clean.lower():
        model_name_clean = model_name_clean.replace("llama", "").strip("-_")
    elif "Qwen" in model_name_clean:
        model_name_clean = model_name_clean.replace("Qwen", "").strip("-_")
    elif "Gemma" in model_name_clean.lower():
        model_name_clean = model_name_clean.replace("gemma", "").strip("-_")


    if not model_name_clean:
        model_name_clean = "custom_model"


    print(f"--- Loading Data for Condition: {condition} ---")

    try:
        _df = None
        file_loaded = False


        if args.add == 'all_extend':
            file_path = os.path.join(DATA_DIR, 'know_bias_data_all_combine_extend.csv')
            if os.path.exists(file_path):
                _df = pd.read_csv(file_path)
                print(f"[OK] Loaded specified dataset: {file_path}")
                file_loaded = True
            else:
                raise FileNotFoundError(f"Specified dataset not found: {file_path}")

        elif args.add == 'extend':
            file_path = os.path.join(DATA_DIR, 'know_bias_data_extend.csv')
            if os.path.exists(file_path):
                _df = pd.read_csv(file_path)
                print(f"[OK] Loaded specified dataset: {file_path}")
                file_loaded = True
            else:
                raise FileNotFoundError(f"Specified dataset not found: {file_path}")


        elif args.add == '' or args.add is None:

            possible_specific_files = [
                os.path.join(DATA_DIR, f"{condition}.csv"),
                os.path.join(DATA_DIR, f"{condition.lower()}.csv"),
                os.path.join(DATA_DIR, f"{condition.replace('_', '')}.csv")
            ]

            for p_file in possible_specific_files:
                if os.path.exists(p_file):
                    print(f"[OK] Found specific condition file: {p_file}")
                    _df = pd.read_csv(p_file)
                    file_loaded = True
                    break


            if not file_loaded:
                fallback_file = os.path.join(DATA_DIR, 'know_bias_data_extend.csv')
                if os.path.exists(fallback_file):
                    print(f"[INFO] Specific file for '{condition}' not found.")
                    print(f"[INFO] Auto-falling back to general dataset: {fallback_file}")
                    _df = pd.read_csv(fallback_file, encoding='utf-8-sig', on_bad_lines='skip')
                    file_loaded = True
                else:

                    available_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(
                        DATA_DIR) else []
                    raise FileNotFoundError(
                        f"Cannot find data for '{condition}'.\n"
                        f"Tried specific files: {possible_specific_files}\n"
                        f"Tried fallback: {fallback_file} (Not found)\n"
                        f"Available files: {available_files}"
                    )

        else:
            raise ValueError(f"Unknown --add argument value: '{args.add}'")

        if not file_loaded or _df is None:
            raise ValueError("Data loading logic failed to assign a dataframe.")

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



    att_scores = []
    att_scores_tmp = []
    group_dic = set()


    print("--- Starting Knowledge Neuron Analysis ---")
    for idx, row in tqdm(_df.iterrows(), total=len(_df), desc="Processing Samples"):

        if idx % 5 == 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        question = row['question']
        target = row['gold']
        q_idx = row['q_index']
        q_polarity = row['q_polarity']



        prompt_file_main = f'{PROMPT_DIR}/evaluate_know_new.txt'
        prompt_file_fallback = f'{PROMPT_DIR}/evaluate_s.txt'

        instruction = ""
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


        if (q_idx, q_polarity) not in group_dic:
            if group_dic:
                print('\n=== GROUP WISE START ===')
                group_neurons_ = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
                print('=== GROUP WISE END ===\n')

                group_dic.add((q_idx, q_polarity))
                print(f'Processing new group: {q_idx}, {q_polarity}')
                att_scores_tmp = []
            else:
                group_dic.add((q_idx, q_polarity))
                print(f'First Processing group: {q_idx}, {q_polarity}')
                att_scores_tmp = []


        delta = kn.search_kns_modi(
            source_prompt,
            target,
            n_prompts=num_prob,
            batch_size=args.qwen_batch_size if is_qwen35 else 4,
            verbose=args.verbose,
            steps=args.steps if is_qwen35 else 20,
            aggressive_cleanup=not is_qwen35,
            validate_finite=not is_qwen35,
        )

        att_score_cpu = delta.detach().cpu()
        att_scores.append(att_score_cpu)
        att_scores_tmp.append(att_score_cpu)


    print('\n=== FINAL GROUP WISE START ===')
    group_neurons = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
    print('=== FINAL GROUP WISE END ===\n')

    print('=== OVERALL EVALUATION START ===')
    all_combined_neurons = kn.get_kns_func(att_scores, threshold=th_val, p=p_val)
    print('=== OVERALL EVALUATION END ===')


    save_filename = (
        f'gender_expected_ans_{args.ans}_'
        f'th{th_val}_p{p_val}_'
        f'{model_name_clean}_'
        f'att_scores_know_bias_new_no_ct{args.add}.npy'
    )
    save_path = os.path.join(save_dir, save_filename)

    print(f"Saving results to: {save_path}")

    np.save(save_path, torch.stack(att_scores).to(torch.float32).cpu().numpy())
    print("Done.")


def know_bias_analysis_contrastive(model_name, condition, num_prob):
    print(f"--- Initializing Model: {model_name} ---")
    kn = load_model(model_name, device_map=get_run_device_map())

    _df = None
    th_val = args.threshold
    p_val = args.p_value


    model_name_clean = model_name
    if '/' in model_name:
        model_name_clean = os.path.basename(model_name)

    if "llama" in model_name_clean.lower():
        model_name_clean = model_name_clean.replace("llama", "").strip("-_")
    elif "Qwen" in model_name_clean:
        model_name_clean = model_name_clean.replace("Qwen", "").strip("-_")

    if not model_name_clean:
        model_name_clean = "custom_model"


    print(f"--- Loading Data for Condition: {condition} ---")

    try:
        _df = None
        file_loaded = False

        if args.add == 'all_extend':
            file_path = os.path.join(DATA_DIR, 'know_bias_data_all_combine_extend.csv')
            if os.path.exists(file_path):
                _df = pd.read_csv(file_path)
                print(f"[OK] Loaded specified dataset: {file_path}")
                file_loaded = True
            else:
                raise FileNotFoundError(f"Specified dataset not found: {file_path}")

        elif args.add == 'extend':
            file_path = os.path.join(DATA_DIR, 'know_bias_data_extend.csv')
            if os.path.exists(file_path):
                _df = pd.read_csv(file_path)
                print(f"[OK] Loaded specified dataset: {file_path}")
                file_loaded = True
            else:
                raise FileNotFoundError(f"Specified dataset not found: {file_path}")

        elif args.add == '' or args.add is None:
            possible_specific_files = [
                os.path.join(DATA_DIR, f"{condition}.csv"),
                os.path.join(DATA_DIR, f"{condition.lower()}.csv"),
                os.path.join(DATA_DIR, f"{condition.replace('_', '')}.csv")
            ]

            for p_file in possible_specific_files:
                if os.path.exists(p_file):
                    print(f"[OK] Found specific condition file: {p_file}")
                    _df = pd.read_csv(p_file)
                    file_loaded = True
                    break

            if not file_loaded:
                fallback_file = os.path.join(DATA_DIR, 'know_bias_data_extend.csv')
                if os.path.exists(fallback_file):
                    print(f"[INFO] Specific file for '{condition}' not found.")
                    print(f"[INFO] Auto-falling back to general dataset: {fallback_file}")
                    _df = pd.read_csv(fallback_file, encoding='utf-8-sig', on_bad_lines='skip')
                    file_loaded = True
                else:
                    available_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
                    raise FileNotFoundError(
                        f"Cannot find data for '{condition}'.\n"
                        f"Tried specific files: {possible_specific_files}\n"
                        f"Tried fallback: {fallback_file} (Not found)\n"
                        f"Available files: {available_files}"
                    )

        else:
            raise ValueError(f"Unknown --add argument value: '{args.add}'")

        if not file_loaded or _df is None:
            raise ValueError("Data loading logic failed to assign a dataframe.")

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

    att_scores = []
    att_scores_tmp = []
    group_dic = set()


    print("--- Starting Contrastive Knowledge Neuron Analysis ---")
    for idx, row in tqdm(_df.iterrows(), total=len(_df), desc="Processing Samples"):

        if idx % 5 == 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        question_1 = row['question']
        question_2 = row['question1']
        q_idx = row['q_index']
        q_polarity = row['q_polarity']


        if (q_idx, q_polarity) not in group_dic:
            if group_dic:
                print('\n=== GROUP WISE START ===')
                group_neurons_ = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
                print('=== GROUP WISE END ===\n')

                group_dic.add((q_idx, q_polarity))
                print(f'Processing new group: {q_idx}, {q_polarity}')
                att_scores_tmp = []
            else:
                group_dic.add((q_idx, q_polarity))
                print(f'First Processing group: {q_idx}, {q_polarity}')
                att_scores_tmp = []



        delta = kn.search_kns_contrastive_questions(
            question_1, question_2,
            target_token_1=None,
            target_token_2=None,
            pbar=True, verbose=args.verbose
        )

        att_score_cpu = delta.detach().cpu()
        att_scores.append(att_score_cpu)
        att_scores_tmp.append(att_score_cpu)


    print('\n=== FINAL GROUP WISE START ===')
    group_neurons = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
    print('=== FINAL GROUP WISE END ===\n')




    print('\n' + '='*60)
    print('激活值差值统计信息（用于确定阈值）')
    print('='*60)

    att_scores_tensor = torch.stack(att_scores).float()


    all_values = att_scores_tensor.flatten().float().clamp(min=0)
    print(f'\n【整体统计】')
    print(f'  样本数: {att_scores_tensor.shape[0]}')
    print(f'  层数: {att_scores_tensor.shape[1]}')
    print(f'  每层神经元数: {att_scores_tensor.shape[2]}')
    print(f'  差值范围: [{all_values.min().item():.6f}, {all_values.max().item():.6f}]')

    print('=== OVERALL EVALUATION START ===')
    if args.top_k > 0:

        print(f'[MODE] Using top_k mode: top {args.top_k}')
        att_scores_tensor = torch.stack(att_scores)
        all_combined_neurons = get_top_k_neurons(
            att_scores_tensor,
            top_k=args.top_k,
        )
        neuron_save_prefix = f"top{args.top_k}"
    else:

        print(f'[MODE] Using threshold mode: {th_val}')
        all_combined_neurons = kn.get_kns_func(att_scores, threshold=th_val, p=p_val)
        neuron_save_prefix = f"th{th_val}"
    print('=== OVERALL EVALUATION END ===')


    contrastive_suffix = f"_contrastive_{args.contrastive_method}"
    save_filename = (
        f'expected_ans_{args.ans}_'
        f'{neuron_save_prefix}_'
        f'{model_name_clean}_'
        f'att_scores_know_bias_new_no_ct{args.add}{contrastive_suffix}.npy'
    )
    save_path = os.path.join(save_dir, save_filename)

    print(f"Saving activation scores to: {save_path}")

    np.save(save_path, torch.stack(att_scores).to(torch.float32).cpu().numpy())


    neuron_csv_filename = (
        f'neurons_'
        f'expected_ans_{args.ans}_'
        f'{neuron_save_prefix}_'
        f'{model_name_clean}_'
        f'know_bias{args.add}{contrastive_suffix}.csv'
    )
    neuron_csv_path = os.path.join(save_dir, neuron_csv_filename)


    neurons_data = []
    for layer_idx, neuron_idx in all_combined_neurons:
        neurons_data.append({'layer': layer_idx, 'neuron': neuron_idx})

    if neurons_data:
        neurons_df = pd.DataFrame(neurons_data)
        neurons_df.to_csv(neuron_csv_path, index=False)
        print(f"Saving neurons to: {neuron_csv_path}")
        print(f"Total neurons found: {len(neurons_data)}")
    else:
        print("No neurons found with the given threshold.")

    print("Done.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--condition', type=str, default='Gender_identity')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--num_probs', '-n', type=int, default=5)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--T', type=str, default="")
    parser.add_argument('--ans', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--kos', type=str, default='show')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--p_value', type=float, default=0.05)
    parser.add_argument('--add', type=str, default='')
    parser.add_argument('--top_k', type=int, default=0,
                        help='直接取前top_k个神经元（如500），默认为0表示使用threshold模式')

    parser.add_argument('--contrastive', action='store_true', help='使用对比方法（gold vs gold1）')
    parser.add_argument('--contrastive_method', type=str, default='direct', choices=['ig', 'direct'],
                        help='对比方法: ig=积分梯度对比, direct=直接激活值相减')
    parser.add_argument('--steps', type=int, default=20, help='积分梯度步数（仅对ig方法有效）')
    parser.add_argument(
        '--qwen_batch_size', type=int, default=4,
        help='Qwen3.5积分梯度微批大小，必须能整除steps；调大可加速但会增加显存',
    )


    global args
    args = parser.parse_args()
    requested_qwen35 = any(
        marker in args.model_name.lower() for marker in ('qwen3.5', 'qwen3_5')
    )
    if requested_qwen35:
        if args.steps <= 0 or args.qwen_batch_size <= 0:
            parser.error('--steps 和 --qwen_batch_size 必须为正整数')
        if args.steps % args.qwen_batch_size != 0:
            parser.error('--steps 必须能被 --qwen_batch_size 整除')


    args.T = strftime('%Y%m%d-%H%M')

    print(args)

    if args.contrastive:
        print("=== Running Contrastive Knowledge Neuron Analysis ===")
        know_bias_analysis_contrastive(args.model_name, args.condition, args.num_probs)
    else:
        know_bias_analysis_temp_all_new(args.model_name, args.condition,
                                        args.num_probs)


if __name__ == '__main__':
    main()


