
import collections
import math
from functools import partial
from typing import Callable, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.tests.frame.methods.test_join import left
from sympy import false
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .patch import *


class KnowledgeNeuronsBase:
    def __init__(
            self,
            model: nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            model_type: str = "bert",
            device: str = None,
    ):
        self.model = model
        self.model_type = model_type






        device_map = model.device
        print("Device Map:", device_map)
        if 'llama' == model_type or 'Qwen' == model_type or 'ministral' == model_type or 'gemma3' == model_type:
            print("full device map (multi-gpus):", model_type)

        self.device = model.device

        self.tokenizer = tokenizer

        self.baseline_activations = None


        config_model_type = self.model.config.model_type.lower() if hasattr(self.model, 'config') and hasattr(
            self.model.config, 'model_type') else ''
        print(f"config.model_type: {config_model_type}")


        if 'gemma3' in config_model_type or 'gemma3' == model_type:
            model_type_name = type(self.model).__name__
            print(f"model type: {model_type_name}")

            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                self.transformer_layers_attr = "model.model.layers"
                self.word_embeddings_attr = "model.model.embed_tokens.weight"
                print("Using path: model.model.layers")
            elif hasattr(self.model, 'layers'):
                self.transformer_layers_attr = "model.layers"
                self.word_embeddings_attr = "model.embed_tokens.weight"
                print("Using path: model.layers")
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
                self.transformer_layers_attr = "transformer.layers"
                self.word_embeddings_attr = "model.embed_tokens.weight"
                print("Using path: transformer.layers")
            else:

                attrs = [a for a in dir(self.model) if not a.startswith('_') and 'layer' in a.lower()]
                raise AttributeError(
                    f"Cannot find transformer layers on {model_type_name}. Available layer-related attrs: {attrs}")
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
        elif self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif 'gptj' == model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.fc_in"
            self.output_ff_attr = "mlp.fc_out.weight"

            self.word_embeddings_attr = "transformer.wte.weight"
        elif 'ministral' == model_type:
            model_class = type(self.model).__name__
            print(f"[Ministral3 Debug] model type: {model_class}")



            path_candidates = (
                ("language_model.model.layers", "language_model.model.embed_tokens.weight"),
                ("model.language_model.layers", "model.language_model.embed_tokens.weight"),
                ("model.language_model.model.layers", "model.language_model.model.embed_tokens.weight"),
                ("model.layers", "model.embed_tokens.weight"),
            )
            for layers_attr, embeddings_attr in path_candidates:
                try:
                    layers = get_attributes(self.model, layers_attr)
                except (AttributeError, KeyError):
                    continue
                if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                    self.transformer_layers_attr = layers_attr
                    self.word_embeddings_attr = embeddings_attr
                    break
            else:
                raise AttributeError(
                    f"Cannot locate Ministral transformer layers on {model_class}; "
                    f"tried: {[item[0] for item in path_candidates]}"
                )
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            print(f"[Ministral3] Using path: {self.transformer_layers_attr}")
        elif 'qwen3.5' == model_type.lower():
            model_class = type(self.model).__name__
            path_candidates = (
                ("model.layers", "model.embed_tokens.weight"),
                ("model.model.layers", "model.model.embed_tokens.weight"),
                ("language_model.model.layers", "language_model.model.embed_tokens.weight"),
                ("model.language_model.layers", "model.language_model.embed_tokens.weight"),
            )
            for layers_attr, embeddings_attr in path_candidates:
                try:
                    layers = get_attributes(self.model, layers_attr)
                except (AttributeError, KeyError):
                    continue
                if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                    self.transformer_layers_attr = layers_attr
                    self.word_embeddings_attr = embeddings_attr
                    break
            else:
                raise AttributeError(
                    f"Cannot locate Qwen3.5 transformer layers on {model_class}; "
                    f"tried: {[item[0] for item in path_candidates]}"
                )
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            print(f"[Qwen3.5] Using path: {self.transformer_layers_attr}")
        elif 'llama' == model_type or 'Qwen' == model_type or 'mistral' == model_type:



            path_candidates = (
                ("model.model.layers", "model.model.embed_tokens.weight"),
                ("model.layers", "model.embed_tokens.weight"),
                ("transformer.layers", "transformer.embed_tokens.weight"),
            )
            for layers_attr, embeddings_attr in path_candidates:
                try:
                    layers = get_attributes(self.model, layers_attr)
                except (AttributeError, KeyError):
                    continue
                if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                    self.transformer_layers_attr = layers_attr
                    self.word_embeddings_attr = embeddings_attr
                    break
            else:
                raise AttributeError(
                    f"Cannot locate transformer layers for {model_type} on "
                    f"{type(self.model).__name__}; tried: "
                    f"{[item[0] for item in path_candidates]}"
                )
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            print(f"[{model_type}] Using path: {self.transformer_layers_attr}")
        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _find_token_position(self, input_ids, target_tokens):
        input_ids_list = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
        target_len = len(target_tokens)

        for i in range(len(input_ids_list) - target_len + 1):
            if input_ids_list[i:i + target_len] == target_tokens:
                return i


        return -1

    def _find_token_position_by_decode(self, input_ids, target_word):
        input_ids_list = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
        for i, tid in enumerate(input_ids_list):
            decoded = self.tokenizer.decode([tid]).strip()
            if decoded == target_word:
                return i
        return -1

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:


            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)


        mask_idx = -1

        if self.model_type == 't5':
            target_input = self.tokenizer(target, return_tensors='pt').to(self.device)
            encoded_input['decoder_input_ids'] = target_input['input_ids']
            encoded_input['decoder_attention_mask'] = target_input['attention_mask']
        if self.model_type == "bert":
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
        elif self.model_type == 't5':
            mask_idx = list(range(encoded_input['decoder_input_ids'].size(1)))
        else:

            mask_idx = -1
        if target is not None:



            if "gpt" in self.model_type or 't5' in self.model_type:
                target = self.tokenizer.encode(target)
            elif any(name in self.model_type.lower() for name in ('llama', 'qwen', 'gemma', 'mistral', 'ministral')):
                target = self.tokenizer.encode(target, add_special_tokens=False)
            else:
                target = self.tokenizer.convert_tokens_to_ids(target)
        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )

        n_sampling_steps = len(target_label) if any(
            name in self.model_type.lower() for name in ('gpt', 'llama', 'qwen', 'gemma', 'mistral', 'ministral')
        ) else 1
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:

                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i] if n_sampling_steps > 1 else target_label



            if self.model_type == 't5':
                for q, target_idx_ in enumerate(target_idx):
                    gt_prob_ = probs[:, q, target_idx_]
                    all_gt_probs.append(gt_prob_)

                    argmax_prob, argmax_id = [i.item() for i in probs[:, q, :].max(dim=-1)]
                    argmax_tokens.append(argmax_id)
                    argmax_str = self.tokenizer.decode([argmax_id])
                    all_argmax_probs.append(argmax_prob)

                    argmax_completion_str += argmax_str
            else:
                gt_prob = probs[:, target_idx]

                all_gt_probs.append(gt_prob)


                argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
                argmax_tokens.append(argmax_id)
                argmax_str = self.tokenizer.decode([argmax_id])
                all_argmax_probs.append(argmax_prob)

                prompt += argmax_str
                argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def generate(self, prompt, n_sampling_steps=1):
        encoded_input, mask_idx, target_label = self._prepare_inputs(prompt, 'good')
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:

                encoded_input, mask_idx, _ = self._prepare_inputs(
                    prompt, 'good'
                )
            outputs = self.model(**encoded_input)
            probs = torch.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            return probs

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self, layer_idx=0):
        ff_layer = self._get_input_ff_layer(layer_idx)
        if hasattr(ff_layer, "out_features"):
            return int(ff_layer.out_features)
        weight = getattr(ff_layer, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) >= 2:
            return int(weight.shape[0])
        raise AttributeError(
            f"Cannot determine intermediate size from layer {layer_idx} "
            f"({type(ff_layer).__name__})"
        )

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        if activations.dim() == 2:
            tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
            return (
                    tiled_activations
                    * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None]
            )
        elif activations.dim() == 3:
            tiled_activations = einops.repeat(activations, "b m d -> (r b) m d", r=steps)
            return (
                    tiled_activations
                    * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None, None]
            )
        else:
            raise Exception(f"Bad!! The dim of Activation is {activations.dim()}")


    def get_baseline_with_activations(
            self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):

        def get_activations(model, layer_idx, mask_idx):

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()


        baseline_activations = self.baseline_activations

        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_layer_all_token_activations(self, encoded_input, layer_idx):
        all_acts = []

        def hook_fn(acts):


            avg_acts = acts.mean(dim=1)
            all_acts.append(avg_acts)

        handle = register_hook(
            self.model,
            layer_idx=layer_idx,
            f=hook_fn,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        _ = self.model(**encoded_input)
        handle.remove()

        return all_acts[0]

    def get_contrastive_activations(
            self,
            prompt_1: str,
            prompt_2: str,
            threshold: float = None,
            adaptive_threshold: float = None,
            percentile: float = None,
            pbar: bool = True,
    ):
        assert (
                sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), "Provide one and only one of threshold / adaptive_threshold / percentile"

        contrastive_scores = []

        for layer_idx in tqdm(
                range(self.n_layers()),
                desc="Getting contrastive activations for each layer...",
                disable=not pbar,
        ):

            if aggressive_cleanup and layer_idx % 5 == 0:
                torch.cuda.empty_cache()


            encoded_input_1, mask_idx_1, _ = self._prepare_inputs(prompt_1)
            _, baseline_activations_1 = self.get_baseline_with_activations(
                encoded_input_1, layer_idx, mask_idx_1
            )


            encoded_input_2, mask_idx_2, _ = self._prepare_inputs(prompt_2)
            _, baseline_activations_2 = self.get_baseline_with_activations(
                encoded_input_2, layer_idx, mask_idx_2
            )



            if baseline_activations_1.shape == baseline_activations_2.shape:
                delta = baseline_activations_1 - baseline_activations_2

                delta = torch.abs(delta)
            else:

                max_bs = max(baseline_activations_1.shape[0], baseline_activations_2.shape[0])
                if baseline_activations_1.shape[0] < max_bs:
                    pad = torch.zeros((max_bs - baseline_activations_1.shape[0],) + baseline_activations_1.shape[1:],
                                      device=baseline_activations_1.device, dtype=baseline_activations_1.dtype)
                    baseline_activations_1 = torch.cat([pad, baseline_activations_1], dim=0)
                if baseline_activations_2.shape[0] < max_bs:
                    pad = torch.zeros((max_bs - baseline_activations_2.shape[0],) + baseline_activations_2.shape[1:],
                                      device=baseline_activations_2.device, dtype=baseline_activations_2.dtype)
                    baseline_activations_2 = torch.cat([pad, baseline_activations_2], dim=0)
                delta = torch.abs(baseline_activations_1 - baseline_activations_2)


            delta = delta.mean(dim=0)

            contrastive_scores.append(delta)


        contrastive_scores = torch.stack(contrastive_scores)


        if adaptive_threshold is not None:
            threshold = contrastive_scores.max().item() * adaptive_threshold
        if threshold is not None:
            neurons = torch.nonzero(contrastive_scores > threshold).cpu().tolist()
            return neurons

        s = contrastive_scores.flatten().detach().cpu().numpy()
        return (
            torch.nonzero(contrastive_scores > np.percentile(s, percentile))
            .cpu()
            .tolist()
        )

    def get_scores(
            self,
            prompt: str,
            ground_truth: str,
            batch_size: int = 10,
            steps: int = 20,
            attribution_method: str = "integrated_grads",
            pbar: bool = True,
            aggressive_cleanup: bool = True,
            validate_finite: bool = True,
    ):

        scores = []

        encoded_input = None

        for layer_idx in tqdm(
                range(self.n_layers()),
                desc="Getting attribution scores for each layer...",
                disable=not pbar,
        ):


            if aggressive_cleanup and layer_idx % 5 == 0:
                torch.cuda.empty_cache()
            layer_scores = self.get_scores_for_layer(
                prompt,
                ground_truth,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
                aggressive_cleanup=aggressive_cleanup,
                validate_finite=validate_finite,
            )

            scores.append(layer_scores)


        scores = [s.to(self.device) for s in scores]
        return torch.stack(scores)

    def get_coarse_neurons(
            self,
            prompt: str,
            ground_truth: str,
            batch_size: int = 10,
            steps: int = 20,
            threshold: float = None,
            adaptive_threshold: float = None,
            percentile: float = None,
            attribution_method: str = "integrated_grads",
            pbar: bool = True,

    ) -> List[List[int]]:
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
        )
        assert (
                sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), "Provide one and only one of threshold / adaptive_threshold / percentile"

        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            coarse_neurons = torch.nonzero(attribution_scores > threshold).cpu().tolist()
            if self.model_type == 't5' and len(coarse_neurons) > 0 and len(coarse_neurons[0]) == 3:
                coarse_neurons = list(set([(layer_idx, neuron_idx) for layer_idx, _, neuron_idx in coarse_neurons]))
            return coarse_neurons
        s = attribution_scores.flatten().detach().cpu().numpy()
        return (
            torch.nonzero(attribution_scores > np.percentile(s, percentile))
            .cpu()
            .tolist()
        )

    def get_refined_neurons(
            self,
            prompts: List[str],
            ground_truth: str,
            negative_examples: Optional[List[str]] = None,
            p: float = 0.8,
            batch_size: int = 10,
            steps: int = 20,
            coarse_adaptive_threshold: Optional[float] = 0.3,
            coarse_threshold: Optional[float] = None,
            coarse_percentile: Optional[float] = None,
            quiet=False,
            refine: bool = False,
    ) -> List[List[int]]:
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = [
            self.get_coarse_neurons(
                prompt,
                ground_truth,
                batch_size=batch_size,
                steps=steps,
                adaptive_threshold=coarse_adaptive_threshold,
                threshold=coarse_threshold,
                percentile=coarse_percentile,
                pbar=False,
            )
            for prompt in tqdm(
                prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
            )
        ]
        if negative_examples is not None:
            negative_neurons = [
                self.get_coarse_neurons(
                    negative_example,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                )
                for negative_example in tqdm(
                    negative_examples,
                    desc="Getting coarse neurons for negative examples",
                    disable=quiet,
                )
            ]
        if not quiet:
            total_coarse_neurons = sum(len(i) for i in coarse_neurons)
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        if refine:
            refined_neurons = [list(neuron) for neuron, count in c.items() if count > t]
        else:
            refined_neurons = [list(neuron) for neuron, count in c.items()]

        if negative_examples is not None and False:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        if not quiet:
            total_refined_neurons = len(refined_neurons)
            print(f"{total_refined_neurons} neurons remaining after refining")
        return refined_neurons

    def get_scores_for_layer(
            self,
            prompt: str,
            ground_truth: str,
            layer_idx: int,
            batch_size: int = 10,
            steps: int = 20,
            encoded_input: Optional[int] = None,
            attribution_method: str = "integrated_grads",
            aggressive_cleanup: bool = True,
            validate_finite: bool = True,
    ):
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        if aggressive_cleanup and torch.cuda.is_available():
            torch.cuda.empty_cache()


        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )


        n_sampling_steps = len(target_label) if any(
            name in self.model_type.lower() for name in ('gpt', 'llama', 'qwen', 'gemma', 'mistral', 'ministral')
        ) else 1
        if attribution_method == "integrated_grads":
            integrated_grads = []

            for i in range(n_sampling_steps):
                if i > 0 and any(
                        name in self.model_type.lower() for name in
                        ('gpt', 'llama', 'qwen', 'gemma', 'mistral', 'ministral')
                ):

                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )

                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)




                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)


                model_dtype = next(self.model.parameters()).dtype
                scaled_weights = scaled_weights.to(model_dtype)


                integrated_grads_this_step = []



                for batch_weights in scaled_weights.chunk(n_batches):

                    model_dtype = next(self.model.parameters()).dtype
                    batch_weights = batch_weights.detach().clone().to(model_dtype).requires_grad_(True)




                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }
                    if self.model_type == "bert":
                        inputs["token_type_ids"] = einops.repeat(
                            encoded_input["token_type_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )
                    if self.model_type == 't5':
                        inputs["decoder_input_ids"] = einops.repeat(
                            encoded_input["decoder_input_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )
                        inputs["decoder_attention_mask"] = einops.repeat(
                            encoded_input["decoder_attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )


                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=mask_idx,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )


                    outputs = self.model(**inputs)




                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1, dtype=torch.float32)











                    if validate_finite:
                        assert torch.isfinite(baseline_activations).all(), "baseline_activations has NaN/Inf"



                        assert torch.isfinite(outputs.logits).all(), "logits has NaN/Inf"
                        assert torch.isfinite(probs).all(), "probs has NaN/Inf"


                    target_idx = (
                        target_label[i] if n_sampling_steps > 1 else target_label
                    )
                    if self.model_type == 't5':
                        assert probs.size(1) == len(target_idx)
                        target_probs = [probs[:, q, target_idx_] for q, target_idx_ in enumerate(target_idx)]

                        grad = torch.autograd.grad(
                            torch.unbind(torch.cat(target_probs, dim=0)), batch_weights
                        )[0]
                        grad = grad.sum(dim=0)
                        integrated_grads_this_step.append(grad)
                    else:
                        grad = torch.autograd.grad(
                            torch.unbind(probs[:, target_idx]), batch_weights
                        )[0]
                        grad = grad.sum(dim=0)


                        if validate_finite:
                            assert torch.isfinite(grad).all(), "grad has NaN/Inf"

                        integrated_grads_this_step.append(grad)



                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    del probs, grad, batch_weights
                    if aggressive_cleanup and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()


                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                integrated_grads.append(integrated_grads_this_step)

                del integrated_grads_this_step, scaled_weights, baseline_activations
                if aggressive_cleanup and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                if n_sampling_steps > 1:
                    prompt += next_token_str
            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
                integrated_grads
            )

            return integrated_grads
        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                if i > 0 and any(
                        name in self.model_type.lower() for name in
                        ('gpt', 'llama', 'qwen', 'gemma', 'mistral', 'ministral')
                ):

                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                activations.append(baseline_activations)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)
                    prompt += next_token_str
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        else:
            raise NotImplementedError

    def erase(self, prompt, neurons):
        _, mask_idx, _ = self._prepare_inputs(prompt, 'good')
        print('mask idx:', mask_idx)

        patch_ff_layer(
            self.model,
            mask_idx,
            mode='suppress',
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )
        modified_layers = list(set([x[0] for x in neurons]))
        return modified_layers

    def erase_0(self, neurons):
        mask_idx = -1

        patch_ff_layer(
            self.model,
            mask_idx,
            mode='suppress',
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )
        modified_layers = list(set([x[0] for x in neurons]))
        return modified_layers

    def enhance(self, neurons, val, neuron_factors=None):
        mask_idx = -1

        patch_ff_layer(
            self.model,
            mask_idx,
            mode='enhance',
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
            _value=val,
            neuron_factors=neuron_factors,

        )
        modified_layers = list(set([x[0] for x in neurons]))
        return modified_layers


class KnowledgeNeurons(KnowledgeNeuronsBase):




















    def search_kns_modi(
            self,
            source,
            target,
            n_prompts=100,
            batch_size=20,
            verbose=False,
            steps=20,
            aggressive_cleanup=True,
            validate_finite=True,
    ):


        att_scores = self.get_scores(
            source,
            target,
            batch_size=batch_size,
            pbar=False,
            steps=steps,
            aggressive_cleanup=aggressive_cleanup,
            validate_finite=validate_finite,
        )


        if verbose:
            print(f"Source: {source}")
            print(f" Bias (gold2): {target}")


        return att_scores.detach()

    def search_kns_contrastive(
            self,
            source: str,
            ground_truth_1: str,
            ground_truth_2: str,
            batch_size: int = 20,
            verbose: bool = False,
            steps: int = 20,
    ):

        att_scores_1 = self.get_scores(
            source, ground_truth_1, batch_size=batch_size, pbar=False, steps=steps
        )

        att_scores_2 = self.get_scores(
            source, ground_truth_2, batch_size=batch_size, pbar=False, steps=steps
        )

        delta = torch.abs(att_scores_1 - att_scores_2)

        if verbose:
            print(f"Source: {source}")
            print(f" Ground truth 1: {ground_truth_1}")
            print(f" Ground truth 2: {ground_truth_2}")
            print(f" Delta shape: {delta.shape}")

        return delta.detach()

    def search_kns_contrastive_direct(
            self,
            source: str,
            ground_truth_1: str,
            ground_truth_2: str,
            pbar: bool = True,
            verbose: bool = False,
    ):
        contrastive_scores = []

        for layer_idx in tqdm(
                range(self.n_layers()),
                desc="Getting contrastive activations for each layer...",
                disable=not pbar,
        ):
            if layer_idx % 5 == 0:
                torch.cuda.empty_cache()


            encoded_input_1, mask_idx_1, target_label_1 = self._prepare_inputs(
                source, ground_truth_1
            )
            _, baseline_activations_1 = self.get_baseline_with_activations(
                encoded_input_1, layer_idx, mask_idx_1
            )


            encoded_input_2, mask_idx_2, target_label_2 = self._prepare_inputs(
                source, ground_truth_2
            )
            _, baseline_activations_2 = self.get_baseline_with_activations(
                encoded_input_2, layer_idx, mask_idx_2
            )


            if layer_idx == 0 and verbose:
                print(f"\n[DEBUG Layer {layer_idx}]")
                print(
                    f"  act1 shape: {baseline_activations_1.shape}, mean: {baseline_activations_1.mean().item():.6f}, max: {baseline_activations_1.abs().max().item():.6f}")
                print(
                    f"  act2 shape: {baseline_activations_2.shape}, mean: {baseline_activations_2.mean().item():.6f}, max: {baseline_activations_2.abs().max().item():.6f}")


            if baseline_activations_1.shape == baseline_activations_2.shape:
                delta = torch.abs(baseline_activations_1 - baseline_activations_2)
            else:

                max_bs = max(baseline_activations_1.shape[0], baseline_activations_2.shape[0])
                if baseline_activations_1.shape[0] < max_bs:
                    pad = torch.zeros((max_bs - baseline_activations_1.shape[0],) + baseline_activations_1.shape[1:],
                                      device=baseline_activations_1.device, dtype=baseline_activations_1.dtype)
                    baseline_activations_1 = torch.cat([baseline_activations_1, pad], dim=0)
                if baseline_activations_2.shape[0] < max_bs:
                    pad = torch.zeros((max_bs - baseline_activations_2.shape[0],) + baseline_activations_2.shape[1:],
                                      device=baseline_activations_2.device, dtype=baseline_activations_2.dtype)
                    baseline_activations_2 = torch.cat([baseline_activations_2, pad], dim=0)
                delta = torch.abs(baseline_activations_1 - baseline_activations_2)


            delta = delta.mean(dim=0)
            contrastive_scores.append(delta)

        contrastive_scores = torch.stack(contrastive_scores)

        if verbose:
            print(f"\n[DEBUG] Source: {source}")
            print(f"  Ground truth 1: {ground_truth_1}")
            print(f"  Ground truth 2: {ground_truth_2}")
            print(f"  Contrastive scores shape: {contrastive_scores.shape}")
            print(
                f"  Contrastive scores range: [{contrastive_scores.min().item():.6f}, {contrastive_scores.max().item():.6f}]")

        return contrastive_scores.detach()

    def search_kns_contrastive_questions(
            self,
            question_1: str,
            question_2: str,
            target_token_1: str = None,
            target_token_2: str = None,
            pbar: bool = True,
            verbose: bool = False,
    ):
        contrastive_scores = []




        import re
        def get_answer_info(text):
            if re.search(r'\byes\b', text, re.IGNORECASE):
                return [10035], "yes"
            elif re.search(r'\bno\b', text, re.IGNORECASE):
                return [912], "no"
            elif "না" in text:
                return self.tokenizer("না", add_special_tokens=False)['input_ids'], "না"
            elif "হ্যাঁ" in text:
                return self.tokenizer("হ্যাঁ", add_special_tokens=False)['input_ids'], "হ্যাঁ"
            else:
                return None, None

        tokens_1, target_word_1 = get_answer_info(question_1) if target_token_1 is None else \
            (self.tokenizer(target_token_1, add_special_tokens=False)['input_ids'] if target_token_1 else [], None)
        tokens_2, target_word_2 = get_answer_info(question_2) if target_token_2 is None else \
            (self.tokenizer(target_token_2, add_special_tokens=False)['input_ids'] if target_token_2 else [], None)

        for layer_idx in tqdm(
                range(self.n_layers()),
                desc="Getting contrastive activations for questions...",
                disable=not pbar,
        ):
            if layer_idx % 5 == 0:
                torch.cuda.empty_cache()


            if verbose and layer_idx == 0:
                print(f"  Q1 original text: {question_1}")
            encoded_input_1, _, _ = self._prepare_inputs(question_1)
            input_ids_1 = encoded_input_1['input_ids'][0]
            if verbose and layer_idx == 0:
                print(f"  Q1 last 10 tokens: {self.tokenizer.decode(input_ids_1[-10:])}")
                print(f"  Q1 last 10 token ids: {input_ids_1[-10:].tolist()}")

            mask_idx_1 = -1
            _, baseline_activations_1 = self.get_baseline_with_activations(
                encoded_input_1, layer_idx, mask_idx_1
            )


            if verbose and layer_idx == 0:
                print(f"  Q2 original text: {question_2}")
            encoded_input_2, _, _ = self._prepare_inputs(question_2)
            input_ids_2 = encoded_input_2['input_ids'][0]
            if verbose and layer_idx == 0:
                print(f"  Q2 last 10 tokens: {self.tokenizer.decode(input_ids_2[-10:])}")
                print(f"  Q2 last 10 token ids: {input_ids_2[-10:].tolist()}")
            mask_idx_2 = -1
            _, baseline_activations_2 = self.get_baseline_with_activations(
                encoded_input_2, layer_idx, mask_idx_2
            )


            if baseline_activations_1.shape == baseline_activations_2.shape:
                delta = torch.abs(baseline_activations_1 - baseline_activations_2)
            else:

                max_bs = max(baseline_activations_1.shape[0], baseline_activations_2.shape[0])
                if baseline_activations_1.shape[0] < max_bs:
                    pad = torch.zeros((max_bs - baseline_activations_1.shape[0],) + baseline_activations_1.shape[1:],
                                      device=baseline_activations_1.device, dtype=baseline_activations_1.dtype)
                    baseline_activations_1 = torch.cat([baseline_activations_1, pad], dim=0)
                if baseline_activations_2.shape[0] < max_bs:
                    pad = torch.zeros((max_bs - baseline_activations_2.shape[0],) + baseline_activations_2.shape[1:],
                                      device=baseline_activations_2.device, dtype=baseline_activations_2.dtype)
                    baseline_activations_2 = torch.cat([baseline_activations_2, pad], dim=0)
                delta = torch.abs(baseline_activations_1 - baseline_activations_2)


            if verbose and layer_idx == 0:
                print(f"\n[DEBUG Layer {layer_idx}]")
                print(f"  Q1 shape: {baseline_activations_1.shape}")
                print(
                    f"  Q1 mean: {baseline_activations_1.mean().item():.6f}, abs_max: {baseline_activations_1.abs().max().item():.6f}")
                print(
                    f"  Q2 mean: {baseline_activations_2.mean().item():.6f}, abs_max: {baseline_activations_2.abs().max().item():.6f}")
                print(f"  Delta mean: {delta.mean().item():.6f}, max: {delta.max().item():.6f}")


            delta = delta.mean(dim=0)
            contrastive_scores.append(delta)

        contrastive_scores = torch.stack(contrastive_scores)

        if verbose:
            print(f"\n[INFO] Contrastive scores shape: {contrastive_scores.shape}")
            print(f"  Range: [{contrastive_scores.min().item():.6f}, {contrastive_scores.max().item():.6f}]")

        return contrastive_scores.detach()

    def search_kns_contrastive_with_generation(
            self,
            source: str,
            ground_truth_1: str,
            ground_truth_2: str,
            pbar: bool = True,
            verbose: bool = False,
    ):
        contrastive_scores = []

        for layer_idx in tqdm(
                range(self.n_layers()),
                desc="Getting contrastive activations with generation...",
                disable=not pbar,
        ):
            if layer_idx % 5 == 0:
                torch.cuda.empty_cache()


            encoded_input_1, mask_idx_1, target_label_1 = self._prepare_inputs(
                source, ground_truth_1
            )
            _, baseline_activations_1 = self.get_baseline_with_activations(
                encoded_input_1, layer_idx, mask_idx_1
            )




            encoded_input_2, mask_idx_2, target_label_2 = self._prepare_inputs(
                source, ground_truth_2
            )
            _, baseline_activations_2 = self.get_baseline_with_activations(
                encoded_input_2, layer_idx, mask_idx_2
            )




            delta = torch.abs(baseline_activations_1 - baseline_activations_2)

            if verbose and layer_idx == 0:
                print(f"\n[DEBUG] Layer {layer_idx}:")
                print(
                    f"  act1 (gt={ground_truth_1}): mean={baseline_activations_1.mean().item():.6f}, abs_max={baseline_activations_1.abs().max().item():.6f}")
                print(
                    f"  act2 (gt={ground_truth_2}): mean={baseline_activations_2.mean().item():.6f}, abs_max={baseline_activations_2.abs().max().item():.6f}")
                print(f"  delta: mean={delta.mean().item():.6f}, max={delta.max().item():.6f}")


            delta = delta.mean(dim=0)
            contrastive_scores.append(delta)

        contrastive_scores = torch.stack(contrastive_scores)

        if verbose:
            print(f"\n[INFO] Contrastive scores shape: {contrastive_scores.shape}")
            print(f"  Range: [{contrastive_scores.min().item():.6f}, {contrastive_scores.max().item():.6f}]")
            print(f"  Note: A zero difference means the intermediate activations match across both forward passes (normal for autoregressive models)")

        return contrastive_scores.detach()


    def get_raw_n(self, att_scores, threshold=0.1):
        if not att_scores:
            print('[WARNING] get_raw_n received no attribution scores')
            return []
        att_scores = torch.stack(att_scores)
        raw_neurons = fix_p_neuron(att_scores, threshold=threshold)
        print('fix p neurons raw: ', raw_neurons)
        return raw_neurons







    def get_kns_func(self, att_scores, threshold=0.1, p=0.05):

        if not att_scores:
            print('[WARNING] get_kns_func received no attribution scores')
            return []
        att_scores = torch.stack(att_scores)
        print('=== START ===')

        group_neurons = fix_p_neuron(att_scores, threshold=threshold, p=p)
        print('fix p neurons: ', group_neurons)
        print(run_kn_alg(att_scores))

        print(cosine_sim_n(att_scores.view(att_scores.shape[0], -1)))
        print('=== END ===')
        return group_neurons

    def get_refined_contrastive_neurons(
            self,
            prompts_1: List[str],
            prompts_2: List[str],
            p: float = 0.8,
            adaptive_threshold: float = 0.3,
            threshold: float = None,
            percentile: float = None,
            quiet: bool = False,
            refine: bool = False,
    ) -> List[List[int]]:
        assert isinstance(prompts_1, list), "Must provide a list of prompts_1"
        assert isinstance(prompts_2, list), "Must provide a list of prompts_2"
        assert len(prompts_1) == len(prompts_2), "prompts_1 and prompts_2 must have the same length"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts_1)
        coarse_neurons = []

        for prompt_1, prompt_2 in tqdm(
                zip(prompts_1, prompts_2), desc="Getting contrastive neurons for each prompt pair...", disable=quiet
        ):
            neurons = self.get_contrastive_activations(
                prompt_1,
                prompt_2,
                adaptive_threshold=adaptive_threshold,
                threshold=threshold,
                percentile=percentile,
                pbar=False,
            )
            coarse_neurons.append(neurons)

        if not quiet:
            total_coarse_neurons = sum(len(i) for i in coarse_neurons)
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")

        t = n_prompts * p
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        if refine:
            refined_neurons = [list(neuron) for neuron, count in c.items() if count > t]
        else:
            refined_neurons = [list(neuron) for neuron, count in c.items()]

        if not quiet:
            total_refined_neurons = len(refined_neurons)
            print(f"{total_refined_neurons} neurons remaining after refining")

        return refined_neurons























    def complete(self, prompts, max_out_len=20, n_gen_per_prompt=1, top_k=5):
        if isinstance(prompts, str):
            prompts = [prompts]
        input_prompts = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
        input_tokens = self.tokenizer(input_prompts, padding=True, return_tensors="pt").to(self.device)

        input_ids, attention_mask = input_tokens["input_ids"], input_tokens["attention_mask"]
        batch_size = input_ids.size(0)
        prompt_size = input_ids.size(1)
        past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

        with torch.no_grad():
            while input_ids.size(1) < max_out_len + prompt_size:
                model_out = self.model(
                    input_ids=input_ids[:, cur_context],
                    attention_mask=attention_mask[:, cur_context],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits, past_key_values = model_out.logits, model_out.past_key_values
                softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)


                tk = torch.topk(softmax_out, top_k, dim=1).indices
                softmax_out_top_k = torch.gather(softmax_out, 1, tk)
                softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
                new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
                new_toks = torch.gather(tk, 1, new_tok_indices)



                if cur_context.stop == input_ids.size(1):
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                    )
                    input_ids = torch.cat(
                        [
                            input_ids,
                            input_ids.new_ones(batch_size, 1) * self.tokenizer.pad_token_id,
                        ],
                        dim=1,
                    )

                last_non_masked = attention_mask.sum(1) - 1
                for i in range(batch_size):
                    new_idx = last_non_masked[i] + 1
                    if last_non_masked[i].item() + 1 != cur_context.stop:
                        continue


                    if new_idx < max_out_len:
                        input_ids[i][new_idx] = new_toks[i]
                        attention_mask[i][new_idx] = 1

                cur_context = slice(cur_context.stop, cur_context.stop + 1)

        text = [self.tokenizer.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
        text = [
            x
            .replace("\n\n", " ")
            .replace("<|endoftext|>", "")
            for x in text
        ]

        return text



def get_refined_neurons(coarse_neurons, p):
    n_samples = len(coarse_neurons)
    t = n_samples * p
    c = collections.Counter()
    for neurons in coarse_neurons:
        for n in neurons:
            c[tuple(n)] += 1

    if p >= 1.0:

        refined_neurons = [list(neuron) for neuron, count in c.items() if count >= n_samples]
    else:

        refined_neurons = [list(neuron) for neuron, count in c.items() if count > t]
    return refined_neurons


def get_top_k_neurons(scores, top_k=500):


    avg_scores = scores.mean(dim=0)


    flat_scores = avg_scores.flatten()
    top_k = min(top_k, flat_scores.numel())
    values, indices = torch.topk(flat_scores, top_k)


    n_layers, hidden_size = avg_scores.shape
    neurons = []
    for idx in indices.cpu().tolist():
        layer_idx = idx // hidden_size
        neuron_idx = idx % hidden_size
        neurons.append([layer_idx, neuron_idx])

    print(f"[INFO] Top {len(neurons)} neurons selected")
    print(f"[INFO] Score range: [{values.min().item():.6f}, {values.max().item():.6f}]")
    return neurons



def fix_p_neuron(scores, threshold=0.1, p=0.05):
    coarse_neurons = []
    for score in scores:
        _threshold = score.max().item() * threshold
        coarse_neurons.append(torch.nonzero(score > _threshold).tolist())

    refined_neurons = get_refined_neurons(coarse_neurons, p)
    return refined_neurons



def run_kn_alg(scores, kn_num_min=2, kn_num_max=5, adaptive_threshold=0.1, p=0.05):
    coarse_neurons = []

    for score in scores:
        threshold = score.max().item() * adaptive_threshold
        coarse_neurons.append(torch.nonzero(score > threshold).tolist())

    for i in range(100):
        refined_neurons = get_refined_neurons(coarse_neurons, p)
        if len(refined_neurons) < kn_num_min:
            p -= 0.05
        elif len(refined_neurons) > kn_num_max:
            p += 0.05
        else:
            break
    return refined_neurons, p



@torch.no_grad()
def cosine_sim_n(scores):

    y = scores / scores.norm(dim=1).unsqueeze(1)
    y = torch.nan_to_num(y, nan=1e-15)

    u, s, v = torch.svd(y.to(torch.float32))
    return (s[0] ** 2 - 1) / (scores.shape[0] - 1)
