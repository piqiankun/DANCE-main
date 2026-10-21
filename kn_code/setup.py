import torch
import os
from huggingface_hub import login
from transformers import (
    BertTokenizer, BertLMHeadModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, LlamaForCausalLM,
    AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, FineGrainedFP8Config,
    Mistral3Config, Ministral3ForCausalLM, Mistral3ForConditionalGeneration,
    Gemma3Config, Gemma3ForCausalLM
)
from .knowledge_neurons import KnowledgeNeurons


def _install_set_submodule_compat():
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def set_submodule(self, target, module, strict=False):
        if not target:
            raise ValueError("Cannot set the root module")
        parent_path, _, child_name = target.rpartition(".")
        parent = self.get_submodule(parent_path) if parent_path else self
        if strict and not hasattr(parent, child_name):
            raise AttributeError(f"{parent_path or '<root>'} has no submodule {child_name}")
        setattr(parent, child_name, module)

    torch.nn.Module.set_submodule = set_submodule



CACHE_DIR = os.getenv('HF_HOME', '.cache/huggingface')
print(f"Current Working Directory: {os.getcwd()}")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception:
        pass


def initialize_model_and_tokenizer(
        model_path: str,
        use_quantization: bool = True,
        device_map=None,
):
    local_files_only = os.path.isdir(model_path)
    model_source = os.path.abspath(model_path) if local_files_only else model_path
    print(f"-> 正在加载模型: {model_source}")
    if use_quantization:
        print("🔥 启用 [4bit量化] 加载模式...")
    elif device_map in ("auto", "balanced", "balanced_low_0", "sequential"):
        print("🔥 启用 [多卡模型并行] 加载模式 (无量化)...")
    else:
        print("🔥 强制启用 [单卡] 加载模式 (无量化)...")



    resolved_device_map = {"": 0} if device_map is None else device_map


    common_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "local_files_only": local_files_only,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "device_map": resolved_device_map,
    }


    if use_quantization:
        common_kwargs["quantization_config"] = bnb_config


    config = None
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
        arch = config.architectures[0] if config.architectures else ""
        is_bert = "BertLMHeadModel" in arch or any(b in model_path.lower() for b in ["bert-base"])
        is_gpt2 = "GPT2LMHeadModel" in arch or any(g in model_path.lower() for g in ["gpt2"])
        is_causal_lm = not is_bert and not is_gpt2
    except Exception:
        is_bert = any(b in model_path.lower() for b in ["bert-base"])
        is_gpt2 = any(g in model_path.lower() for g in ["gpt2"])
        is_causal_lm = True


    if is_bert:
        print("检测到 BERT 架构...")
        tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        model = BertLMHeadModel.from_pretrained(**common_kwargs)
    elif is_gpt2:
        print("检测到 GPT-2 架构...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        model = GPT2LMHeadModel.from_pretrained(**common_kwargs)
    elif is_causal_lm:
        print("检测到 Causal LM 架构 (Llama/Qwen/Gemma/Ministral)...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            rope_scaling=None,
            local_files_only=local_files_only,
        )



        config_model_type = getattr(config, "model_type", "").lower()
        if config_model_type == "qwen3_5":
            print("检测到 Qwen3.5 多模态架构，加载纯文本 causal LM...")
            model = AutoModelForCausalLM.from_pretrained(**common_kwargs)

        elif config_model_type == "qwen3_5_moe":
            raise NotImplementedError(
                "Qwen3.5-MoE 的专家层不能直接使用 dense 模型的 (layer, neuron) 坐标；"
                "请先为 expert 路由定义神经元坐标。"
            )


        elif isinstance(config, Mistral3Config):



            print("检测到 Ministral3 多模态架构，加载官方完整模型并启用文本模式...")
            _install_set_submodule_compat()
            ministral_kwargs = common_kwargs.copy()



            ministral_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
            full_model = Mistral3ForConditionalGeneration.from_pretrained(**ministral_kwargs)




            model = full_model

        elif False and isinstance(config, Mistral3Config):
            print("检测到 Ministral3 多模态架构，手动加载并重映射权重...")


            from safetensors.torch import load_file


            weight_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
            if not weight_files:
                weight_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]

            print(f"找到权重文件: {weight_files}")


            full_state_dict = {}
            for wf in weight_files:
                wf_path = os.path.join(model_path, wf)
                if wf.endswith('.safetensors'):
                    sd = load_file(wf_path)
                else:
                    sd = torch.load(wf_path, map_location='cpu')
                full_state_dict.update(sd)

            print(f"共加载 {len(full_state_dict)} 个权重")


            sample_keys = list(full_state_dict.keys())[:5]
            print("原始权重key示例:")
            for k in sample_keys:
                print(f"  {k}")








            new_state_dict = {}
            remapped_count = 0
            for key, value in full_state_dict.items():
                if key.startswith('language_model.model.'):


                    new_key = key.replace('language_model.model.', 'model.', 1)
                    new_state_dict[new_key] = value
                    remapped_count += 1
                elif key.startswith('language_model.'):

                    new_key = key.replace('language_model.', 'model.', 1)
                    new_state_dict[new_key] = value
                    remapped_count += 1
                else:
                    new_state_dict[key] = value

            print(f"重映射了 {remapped_count} 个权重key")
            print("重映射后的key示例:")
            for i, k in enumerate(list(new_state_dict.keys())[:5]):
                print(f"  {k}")



            def set_attr(cfg, name, value):
                if not hasattr(cfg, name) or getattr(cfg, name) is None:
                    setattr(cfg, name, value)
                    return True
                return False



            sample_embed = None
            for key in new_state_dict:
                if 'embed_tokens' in key and 'weight' in key:
                    sample_embed = new_state_dict[key]
                    break

            vocab_size = sample_embed.shape[0] if sample_embed is not None else 131072
            hidden_size = sample_embed.shape[1] if sample_embed is not None else 3072

            set_attr(config, 'vocab_size', vocab_size)
            set_attr(config, 'hidden_size', hidden_size)
            set_attr(config, 'intermediate_size', 8192)
            set_attr(config, 'num_hidden_layers', 26)
            set_attr(config, 'num_attention_heads', 24)
            set_attr(config, 'num_key_value_heads', 2)
            set_attr(config, 'hidden_act', 'silu')
            set_attr(config, 'max_position_embeddings', 32768)
            set_attr(config, 'rms_norm_eps', 1e-5)
            set_attr(config, 'rope_theta', 10000.0)
            set_attr(config, 'sliding_window', 4096)
            set_attr(config, 'attention_dropout', 0.0)
            set_attr(config, 'hidden_dropout', 0.0)
            set_attr(config, 'pad_token_id', 0)
            set_attr(config, 'model_type', 'ministral3')
            set_attr(config, 'rope_parameters', {'rope_type': 'default'})
            set_attr(config, 'tie_word_embeddings', False)

            print(f"Config设置: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")

            print("使用config创建模型...")
            full_model = Ministral3ForCausalLM(config)
            print(f"模型结构创建成功: {type(full_model).__name__}")

            print(f"模型结构创建成功: {type(full_model).__name__}")



            if hasattr(full_model, 'lm_head') and hasattr(full_model.model, 'embed_tokens'):
                print("解开 lm_head 和 embed_tokens 的绑定...")

                embed_tokens_weight = full_model.model.embed_tokens.weight

                full_model.lm_head.weight = torch.nn.Parameter(embed_tokens_weight.clone().detach())


            missing, unexpected = full_model.load_state_dict(new_state_dict, strict=False)
            print(f"权重加载结果: missing={len(missing)}, unexpected={len(unexpected)}")


            if 'lm_head.weight' in missing:
                print("lm_head.weight 缺失，从 embed_tokens 复制...")
                full_model.lm_head.weight = torch.nn.Parameter(full_model.model.embed_tokens.weight.clone().detach())

            if missing:
                print("MISSING 权重 (未能加载):")
                for m in missing[:5]:
                    print(f"  {m}")
                if len(missing) > 5:
                    print(f"  ... 共 {len(missing)} 个")


            embed_weight = full_model.model.embed_tokens.weight
            lm_head_weight = full_model.lm_head.weight


            embed_mean = embed_weight.mean().item()
            embed_std = embed_weight.std().item()
            is_embed_random = abs(embed_mean) < 0.01 and embed_std < 0.1

            print(f"embed_tokens: mean={embed_mean:.6f}, std={embed_std:.6f}, 看起来{'随机' if is_embed_random else '已加载'}")
            print(f"lm_head: mean={lm_head_weight.mean().item():.6f}, std={lm_head_weight.std().item():.6f}")


            if hasattr(full_model.model, 'layers') and len(full_model.model.layers) > 0:
                layer0 = full_model.model.layers[0]
                if hasattr(layer0, 'mlp'):
                    gate_proj = layer0.mlp.gate_proj.weight
                    print(f"layer0.mlp.gate_proj: mean={gate_proj.mean().item():.6f}, std={gate_proj.std().item():.6f}")

            has_embed = hasattr(full_model.model, 'embed_tokens') and full_model.model.embed_tokens.weight.numel() > 0
            has_lm_head = hasattr(full_model, 'lm_head') and full_model.lm_head.weight.numel() > 0
            has_layers = hasattr(full_model.model, 'layers') and len(full_model.model.layers) > 0

            print(f"组件检查: embed_tokens={'OK' if has_embed else 'EMPTY'}, lm_head={'OK' if has_lm_head else 'EMPTY'}, layers={'OK' if has_layers else 'EMPTY'}")


            print("✅ 使用 full_model (Ministral3ForCausalLM)")
            model = full_model

        elif isinstance(config, Gemma3Config):
            print("检测到 Gemma3 架构，使用专用类加载...")
            model_kwargs = common_kwargs.copy()
            model_kwargs["trust_remote_code"] = True
            model = Gemma3ForCausalLM.from_pretrained(**model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(**common_kwargs)
    else:
        raise ValueError(f"无法识别模型架构: {model_path}")



    if hasattr(model, 'config'):
        model.config.use_cache = False


    for param in model.parameters():
        param.requires_grad = False


    model.eval()
    return model, tokenizer


def model_type(model_path: str):
    path_lower = model_path.lower()
    if 'bert' in path_lower:
        return "bert"
    elif 'gpt2' in path_lower:
        return "gpt2"
    elif 'llama' in path_lower:
        return 'llama'
    elif 'qwen3.5' in path_lower or 'qwen3_5' in path_lower:
        return 'Qwen3.5'
    elif 'qwen' in path_lower:
        return 'Qwen'
    elif 'ministral' in path_lower:
        return 'ministral'
    elif 'gemma-3' in path_lower or 'gemma3' in path_lower:
        return 'gemma3'
    else:
        try:
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            arch = config.architectures[0] if config.architectures else ""
            if "Llama" in arch: return 'llama'
            if "Qwen3_5" in arch: return 'Qwen3.5'
            if "Qwen" in arch: return 'Qwen'
            if "Ministral" in arch: return 'ministral'
            if "Gemma3" in arch: return 'gemma3'
        except:
            pass
        return 'llama'


def load_model(
        model_path,
        device=None,
        use_quantization: bool = False,
        device_map=None,
):
    print(f"开始加载模型: {model_path}")
    model, tokenizer = initialize_model_and_tokenizer(
        model_path,
        use_quantization=use_quantization,
        device_map=device_map,
    )
    m_type = model_type(model_path)


    kn = KnowledgeNeurons(model, tokenizer, model_type=m_type)

    if m_type in ['gpt2', 'llama', 'Qwen', 'Qwen3.5', 'gemma', 'gemma3', 'mistral', 'ministral']:
        if kn.tokenizer.pad_token is None:
            kn.tokenizer.pad_token = kn.tokenizer.eos_token
            print("Set pad_token to eos_token.")

    print("模型加载完成！")
    return kn
