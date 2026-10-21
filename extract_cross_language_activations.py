
import json
import argparse
import torch
import os

from kn_code import load_model


DATA_FILE = "data/cross_language.jsonl"
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


def get_last_token_activations(kn, sentence: str, neurons: list):
    model = kn.model
    tokenizer = kn.tokenizer


    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]


    last_token_pos = seq_len - 1


    neurons_by_layer = {}
    for layer_idx, neuron_idx in neurons:
        if layer_idx not in neurons_by_layer:
            neurons_by_layer[layer_idx] = []
        neurons_by_layer[layer_idx].append(neuron_idx)


    activations = {neuron: None for neuron in neurons}


    def make_hook_fn(target_layer_idx, target_neuron_idx):
        def hook_fn(module, input, output):

            activations[(target_layer_idx, target_neuron_idx)] = output[0, last_token_pos, target_neuron_idx].detach().cpu().item()
        return hook_fn




    handles = []
    transformer_layers = kn._get_transformer_layers()
    for layer_idx, neuron_indices in neurons_by_layer.items():
        up_proj = transformer_layers[layer_idx].mlp.up_proj
        for neuron_idx in neuron_indices:
            hook_fn = make_hook_fn(layer_idx, neuron_idx)
            handle = up_proj.register_forward_hook(hook_fn)
            handles.append(handle)

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return activations


def main():
    parser = argparse.ArgumentParser(description="获取bn_sentence最后一个token的神经元激活值")
    parser.add_argument("--model", "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="模型名称")
    parser.add_argument("--input_file", type=str, default=DATA_FILE,
                        help="输入JSONL文件路径")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出JSONL文件路径（默认在输入文件后加.activation后缀）")
    parser.add_argument("--start", type=int, default=0,
                        help="起始索引")
    parser.add_argument("--end", type=int, default=-1,
                        help="结束索引，-1表示处理到文件末尾")

    args = parser.parse_args()


    print(f"--- Loading Model: {args.model} ---")
    kn = load_model(args.model, device_map=get_cross_language_device_map())
    model = kn.model
    tokenizer = kn.tokenizer

    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        dispatched_devices = sorted(
            {str(device) for device in hf_device_map.values()},
            key=str,
        )
        print(
            "Accelerate model dispatch devices: "
            f"{dispatched_devices} ({len(hf_device_map)} mapped modules)"
        )


    n_layers = kn.n_layers()
    neurons_to_analyze = SPECIFIED_NEURONS

    layer_intermediate_sizes = {}
    for layer_idx, neuron_idx in neurons_to_analyze:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"层索引 {layer_idx} 超出范围 [0, {n_layers - 1}]")
        if layer_idx not in layer_intermediate_sizes:
            layer_intermediate_sizes[layer_idx] = kn.intermediate_size(layer_idx)
        intermediate_size = layer_intermediate_sizes[layer_idx]
        if neuron_idx < 0 or neuron_idx >= intermediate_size:
            raise ValueError(
                f"第 {layer_idx} 层神经元索引 {neuron_idx} 超出范围 "
                f"[0, {intermediate_size - 1}]"
            )

    print(f"Model: {args.model}")
    print(
        f"Total layers: {n_layers}, Intermediate sizes: "
        f"{sorted(set(layer_intermediate_sizes.values()))}"
    )
    print(f"指定分析的神经元数量: {len(neurons_to_analyze)}")


    print(f"--- Loading Data from: {args.input_file} ---")
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    total = len(lines)

    end_idx = args.end if args.end != -1 else total
    lines = lines[args.start:end_idx]

    print(f"Processing {len(lines)} samples (index {args.start} to {end_idx-1})")

    for i, line in enumerate(lines):
        entry = json.loads(line.strip())
        bn_sentence = entry.get("bn_sentence", "")

        if not bn_sentence:
            print(f"[{i+1}] Warning: Empty bn_sentence for uuid {entry.get('uuid', 'unknown')}")
            entry["activations"] = {str(neuron): None for neuron in neurons_to_analyze}
            results.append(entry)
            continue

        try:
            activations = get_last_token_activations(
                kn, bn_sentence, neurons_to_analyze
            )

            entry["activations"] = {str(k): v for k, v in activations.items()}

            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(lines)}] bn_sentence: {bn_sentence[:30]}...")

        except Exception as e:
            print(f"[{i+1}] Error processing: {e}")
            entry["activations"] = {str(neuron): None for neuron in neurons_to_analyze}
            entry["error"] = str(e)

        results.append(entry)


    output_file = args.output_file
    if output_file is None:
        os.makedirs("results/", exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(args.input_file))
        output_file = f"results/{base}.activations{ext}"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 处理完成，共 {len(results)} 条，输出至: {output_file}")


if __name__ == "__main__":
    main()


