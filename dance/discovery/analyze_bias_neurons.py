from dance.paths import DATA_DIR as PROJECT_DATA_DIR, PROMPT_DIR as PROJECT_PROMPT_DIR, RESULTS_DIR as PROJECT_RESULTS_DIR

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import ttest_ind
from datasets import load_dataset

from dance.core.kn_code.patch import unpatch_ff_layers
from dance.core.kn_code import load_model
from time import strftime
import argparse
import ast
import os
import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable



save_dir = PROJECT_RESULTS_DIR
os.makedirs(save_dir, exist_ok=True)
PROMPT_DIR = PROJECT_PROMPT_DIR
att_score_dir = PROJECT_RESULTS_DIR

os.makedirs(att_score_dir, exist_ok=True)


def get_analysis_device_map():
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpus >= 2:
        print(
            f"Detected {visible_gpus} visible GPUs; "
            "using device_map='balanced' for analysis model parallelism."
        )
        return "balanced"

    print(
        f"Detected {visible_gpus} visible GPU(s); "
        "using the original single-device loading."
    )
    return None



def know_bias_analysis_temp_all_new(model_name, condition, num_prob):
    kn = load_model(model_name, device_map=get_analysis_device_map())

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
    _df = None

    DATA_DIR = PROJECT_DATA_DIR

    th_val = args.threshold
    p_val = args.p_value


    if args.add == 'all_extend':
        _df = pd.read_csv(os.path.join(DATA_DIR, 'know_bias_data_all_combine_extend.csv'))
    elif args.add == 'extend':
        _df = pd.read_csv(os.path.join(DATA_DIR, 'know_bias_data_extend.csv'))
    elif args.add == '':
        default_file = 'know_bias_data_extend.csv'
        print(f"[OK] Using default dataset: {default_file}")
        _df = pd.read_csv(os.path.join(DATA_DIR, default_file))
    else:
        raise ValueError(f"Unknown --add argument: {args.add!r}")

    att_scores = []
    att_scores_tmp = []
    group_dic = set()
    table = PrettyTable()
    table_detail = PrettyTable()

    table.field_names = ['bias cat', 'bias res', 'q_index', 'q_polarity', 'group neurons']
    table_detail.field_names = ['idx','question', 'context', 'q_index', 'q_polarity', 'bias cat', 'bias res', 'raw neurons']


    att_scores_all = np.array([])
    att_scores_path = None
    if args.ans == 0:
        model_name_lower = args.model_name.lower()

        if args.add == '':
            if 'qwen' in model_name_lower:

                att_scores_path = os.path.join(
                    att_score_dir,
                    f'religion_expected_ans_0_th0.1_p0.05_3.5-27B_att_scores_know_bias_new_no_ct.npy'
                )
        elif args.add == 'all':
            if '3.1-8b' in model_name_lower:
                att_scores_path = os.path.join(att_score_dir, 'expected_ans_0_th0.1_p0.05_3.1-8B_att_scores_know_bias_new_no_ct.npy')
        elif args.add == 'extend':
            if '3.2-3b' in model_name_lower:
                att_scores_path = os.path.join(att_score_dir, 'expected_ans_0_0.1_0.49_Llama-3.2-3B-Instruct_att_scores_know_bias_new_n_ctextend.npy')
        else:
            if '3.2-3b' in model_name_lower:
                att_scores_path = os.path.join(att_score_dir, 'expected_ans_0_0.1_0.49_llama3_2_3B_att_scores_know_bias_new.npy')

    if att_scores_path is None:
        raise ValueError(
            f'No attribution-score configuration matches the model/--add/--ans settings: '
            f'model={args.model_name}, add={args.add!r}, ans={args.ans}'
        )
    if not os.path.exists(att_scores_path):
        raise FileNotFoundError(f'Attribution-score file not found: {att_scores_path}')

    print(f'[OK] Loading attribution scores: {att_scores_path}')
    att_scores_all = np.load(att_scores_path, allow_pickle=True)
    if len(att_scores_all) == 0:
        raise ValueError(f'Attribution-score file is empty: {att_scores_path}')
    if len(att_scores_all) < len(_df):
        raise ValueError(
            f'Attribution-score count {len(att_scores_all)} is smaller than the data row count {len(_df)}: '
            f'{att_scores_path}'
        )


    att_scores_tensor = torch.tensor(att_scores_all)

    for idx, row in tqdm(_df.iterrows(), total=len(_df)):
        question = row['question']
        q_idx = row['q_index']
        q_polarity = row['q_polarity']
        bias_cat = row['bias_cat']


        if (q_idx, q_polarity) not in group_dic:
            if group_dic:
                print('=== GROUP WISE START ===')
                group_neurons_ = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
                print('=== GROUP WISE END ===')
                table.add_row([prev_bias_cat, 'na', prev_q_idx, prev_q_polarity, group_neurons_])

                group_dic.add((q_idx, q_polarity))
                print(f'Processing group: {q_idx}, {q_polarity}')
                prev_q_idx = q_idx
                prev_q_polarity = q_polarity
                prev_bias_cat = bias_cat


                att_scores_tmp = []
            else:
                group_dic.add((q_idx, q_polarity))
                print(f'First Processing group: {q_idx}, {q_polarity}')
                prev_q_idx = q_idx
                prev_q_polarity = q_polarity
                prev_bias_cat = bias_cat


                att_scores_tmp = []


        if att_scores_tensor.size(0) == 0:
            print("Skipping an empty knowledge-neuron group")
            continue
        att_score = att_scores_tensor[idx]
        att_scores.append(att_score)
        att_scores_tmp.append(att_score)

        print('L188 individual neurons: ', idx)
        raw_neurons = kn.get_raw_n([att_score], threshold=th_val)
        table_detail.add_row([idx, question, 'na', q_idx, q_polarity, bias_cat, 'na', raw_neurons])


    print('=== GROUP WISE START ===')
    group_neurons = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
    print('=== GROUP WISE END ===')
    table.add_row([prev_bias_cat, 'na', prev_q_idx, prev_q_polarity, group_neurons])

    print('=== OVERAL EVAL ===')
    all_combined = kn.get_kns_func(att_scores, threshold=th_val, p=p_val)

    concept_dir = os.path.join('neuron_res')
    os.makedirs(concept_dir, exist_ok=True)

    if "llama" in args.model_name.lower():
        model_name = args.model_name.split('/')[-1]
    elif "Qwen" in args.model_name:
        model_name = args.model_name.split('/')[-1]
    elif "mistral" in args.model_name.lower():
        model_name = args.model_name.split('/')[-1]
    elif 'gemma' in args.model_name.lower():
        model_name = args.model_name.split('/')[-1]
    else:
        model_name = args.model_name.split('/')[-1]

    save_1 = pd.DataFrame(table._rows, columns=table.field_names)
    save_2 = pd.DataFrame(table_detail._rows, columns=table_detail.field_names)

    save_concepts_dir1 = os.path.join(concept_dir, model_name + '_ans_' + str(args.ans) + '_' + str(th_val) + '_' + str(p_val) +'_group_res_know_bias_' + args.add + 'o_no_ct_tt.csv')

    save_1.to_csv(save_concepts_dir1, index = False, header=True)
    save_concepts_dir2 = os.path.join(concept_dir, model_name + '_ans_' + str(args.ans) + '_' + str(th_val) + '_' + str(p_val) + '_detail_res_know_bias_' + args.add + 'o_no_ct_tt.csv')
    save_2.to_csv(save_concepts_dir2, index = False, header=True)



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
    parser.add_argument('--end', type=int, default=60)
    parser.add_argument('--kos', type=str, default='show')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--p_value', type=float, default=0.05)
    parser.add_argument('--add', type=str, default='')

    global args
    args = parser.parse_args()


    args.T = strftime('%Y%m%d-%H%M')

    print(args)

    know_bias_analysis_temp_all_new(args.model_name, args.condition, args.num_probs)


if __name__ == '__main__':
    main()


