
from kn_code import load_model

import os

CACHE_DIR = os.getenv('HF_HOME', '.cache/huggingface')


os.environ['HF_HOME'] = CACHE_DIR
import random
import csv

import argparse
import torch
import itertools
from transformers import GenerationConfig, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import MllamaForConditionalGeneration, MllamaForCausalLM, AutoProcessor, FbgemmFp8Config

import openai
import time
from typing import *
import json
import re
import copy
import math
import datetime

from torch.nn import functional as F




import numpy as np

from openai import RateLimitError, Timeout, APIError, APIConnectionError, OpenAIError, AzureOpenAI, OpenAI

import base64

import logging

from huggingface_hub import login

from kn_code.patch import Patch, unpatch_ff_layers










HF_TOKEN = os.getenv("HF_TOKEN")

PROMPT_DIR = 'prompt_instructions'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)



class LLM_KN:



    def __init__(
            self,
            model_name,
            load_in_8bit=False,
            load_in_4bit=False,
            device_map=None,
            max_new_tokens=30,
            temperature=0.3,
            repetition_penalty=1.2,
            top_p=1,
            top_k=50,
            do_sample=True,
            cache_dir=CACHE_DIR,
            gpu=0,
            verbose=False,
            train_mode=False,

    ):
        vars = locals()
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)

        self.kn = load_model(model_name, device_map=device_map)

        hf_device_map = getattr(self.kn.model, "hf_device_map", None)
        if hf_device_map:
            dispatched_devices = sorted(
                {str(device) for device in hf_device_map.values()},
                key=str,
            )
            print(
                "Accelerate model dispatch devices: "
                f"{dispatched_devices} ({len(hf_device_map)} mapped modules)"
            )

        print("Loaded base LLM here: ", model_name)
        self.model_name = model_name

    def _is_qwen35(self):
        model_name = self.model_name.lower()
        return "qwen3.5" in model_name or "qwen3_5" in model_name

    def _apply_chat_template(self, messages):
        template_kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if self._is_qwen35():
            template_kwargs["enable_thinking"] = False
        return self.kn.tokenizer.apply_chat_template(
            messages, **template_kwargs
        ).to(self.kn.model.device)

    def getOutput(self, prompt, message=None):

        if 'gemma-2' in self.model_name:
            print('load gemma-2, L486')
            m = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            m = [
                {
                    "role": "system",
                    "content": "You are a useful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        if message is not None:
            m = message

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if 'Instruct' in self.model_name or 'chat' in self.model_name or self._is_qwen35():
            return_output = self._apply_chat_template(m)
        else:
            return_output = self.kn.tokenizer(prompt, return_tensors="pt").to(self.kn.model.device)

        output = self.kn.model.generate(**return_output, generation_config=generation_config)

        if "Llama-2" in self.model_name:
            response = self.kn.tokenizer.decode(output[0]).split("[/INST]")[1].split("</s>")[0].strip()
        elif "Llama-3" in self.model_name:
            prompt_length = return_output["input_ids"].shape[1]
            generated_tokens = output[:, prompt_length:]
            if 'Instruct' in self.model_name:
                response = self.kn.tokenizer.decode(generated_tokens[0]).split("<|eot_id|>")[0].strip()
            else:
                response = self.kn.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        elif 'Qwen' in self.model_name or 'gemma' in self.model_name:
            prompt_length = return_output["input_ids"].shape[1]
            generated_tokens = output[:, prompt_length:]
            response = self.kn.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        else:

            prompt_length = return_output["input_ids"].shape[1]
            generated_tokens = output[:, prompt_length:]
            response = self.kn.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

        if self.verbose:
            print("### RESPONSE ###")
            print(response)

        return response

    def getProb(self, prompt, ans, message=None, return_log=False):
        if 'gemma-2' in self.model_name:
            m = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            m = [
                {
                    "role": "system",
                    "content": "You are a useful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        if message is not None:
            m = message

        self.kn.model.eval()

        def incremental_(prompt, answer):
            device = self.kn.model.device
            prompt_ids = self._apply_chat_template(prompt)
            answer_ids = self.kn.tokenizer.encode(answer, return_tensors="pt", add_special_tokens=False)[0].to(device)

            log_probs_sum = 0.0
            total_tokens = len(answer_ids)

            context_ids = prompt_ids.input_ids[0]





            use_kv_cache = False

            if use_kv_cache:

                input_ids = prompt_ids.input_ids

                with torch.no_grad():
                    outputs = self.kn.model(input_ids, use_cache=True)
                    past_key_values = outputs.past_key_values

                    next_token_logits = outputs.logits[0, -1, :]
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                token_log_prob = next_token_log_probs[answer_ids[0]].item()
                log_probs_sum += token_log_prob

                for i in range(1, total_tokens):
                    token_id = answer_ids[i].unsqueeze(0).unsqueeze(0)

                    with torch.no_grad():
                        outputs = self.kn.model(
                            token_id,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                        past_key_values = outputs.past_key_values

                        next_token_logits = outputs.logits[0, -1, :]
                        next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                    token_log_prob = next_token_log_probs[token_id[0, 0]].item()
                    log_probs_sum += token_log_prob
            else:

                for i, token_id in enumerate(answer_ids):
                    input_ids = context_ids.unsqueeze(0)

                    with torch.no_grad():
                        outputs = self.kn.model(input_ids)

                        next_token_logits = outputs.logits[0, -1, :]

                        next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                    token_log_prob = next_token_log_probs[token_id].item()
                    log_probs_sum += token_log_prob

                    context_ids = torch.cat([context_ids, torch.tensor([token_id]).to(device)], dim=0)




            if self._is_qwen35():
                return log_probs_sum / max(total_tokens, 1)
            return log_probs_sum

        prob = incremental_(m, ans)
        if return_log:
            return prob
        return math.exp(prob)

    def getLogProb_pairs_batch(self, prompts, answers, batch_size=4):
        if len(prompts) != len(answers):
            raise ValueError("prompts and answers must have the same length")
        if not prompts:
            return []

        device = self.kn.model.device
        active_patches = [
            module for module in self.kn.model.modules()
            if isinstance(module, Patch)
        ]
        results = []

        for chunk_start in range(0, len(prompts), batch_size):
            chunk_prompts = prompts[chunk_start:chunk_start + batch_size]
            chunk_answers = answers[chunk_start:chunk_start + batch_size]
            prompt_token_lists = []
            answer_token_lists = []

            for prompt, answer in zip(chunk_prompts, chunk_answers):
                messages = [
                    {"role": "system", "content": "You are a useful assistant."},
                    {"role": "user", "content": prompt},
                ]
                encoded_prompt = self._apply_chat_template(messages)
                prompt_token_lists.append(encoded_prompt.input_ids[0].tolist())
                answer_token_lists.append(
                    self.kn.tokenizer.encode(answer, add_special_tokens=False)
                )

            if any(len(ids) == 0 for ids in answer_token_lists):
                raise ValueError("Cannot score an empty answer")

            rows = [
                prompt_ids + answer_ids[:-1]
                for prompt_ids, answer_ids in zip(prompt_token_lists, answer_token_lists)
            ]
            max_len = max(len(row) for row in rows)
            pad_token_id = self.kn.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.kn.tokenizer.eos_token_id or 0
            input_ids = torch.full(
                (len(rows), max_len), pad_token_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(rows), max_len), dtype=torch.long, device=device
            )
            scoring_mask = torch.zeros(
                (len(rows), max_len), dtype=torch.bool, device=device
            )

            for row_idx, (row, prompt_ids, answer_ids) in enumerate(
                zip(rows, prompt_token_lists, answer_token_lists)
            ):
                input_ids[row_idx, :len(row)] = torch.as_tensor(
                    row, dtype=torch.long, device=device
                )
                attention_mask[row_idx, :len(row)] = 1
                score_start = len(prompt_ids) - 1
                scoring_mask[
                    row_idx, score_start:score_start + len(answer_ids)
                ] = True

            for patch_module in active_patches:
                patch_module.runtime_position_mask = scoring_mask
            try:
                with torch.inference_mode():
                    outputs = self.kn.model(input_ids, attention_mask=attention_mask)
                    for row_idx, (prompt_ids, answer_ids) in enumerate(
                        zip(prompt_token_lists, answer_token_lists)
                    ):
                        score_start = len(prompt_ids) - 1
                        positions = torch.arange(
                            score_start,
                            score_start + len(answer_ids),
                            device=device,
                        )
                        targets = torch.as_tensor(
                            answer_ids, dtype=torch.long, device=device
                        )
                        logits = outputs.logits[row_idx, positions, :]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        selected = log_probs[
                            torch.arange(len(answer_ids), device=device), targets
                        ]
                        score = selected.sum().item()
                        if self._is_qwen35():
                            score /= len(answer_ids)
                        results.append(score)
            finally:
                for patch_module in active_patches:
                    patch_module.runtime_position_mask = None

        return results

    def getProb_batch(self, prompt, answers, message=None, correct_causal_alignment=False):
        if 'gemma-2' in self.model_name:
            m = [{"role": "user", "content": prompt}]
        else:
            m = [
                {"role": "system", "content": "You are a useful assistant."},
                {"role": "user", "content": prompt}
            ]

        if message is not None:
            m = message

        self.kn.model.eval()


        prompt_ids = self._apply_chat_template(m)
        prompt_len = prompt_ids.input_ids.shape[1]


        answer_ids_list = []
        for ans in answers:
            ans_ids = self.kn.tokenizer.encode(ans, add_special_tokens=False)
            answer_ids_list.append(ans_ids)



        active_patches = [
            module for module in self.kn.model.modules()
            if isinstance(module, Patch)
        ]
        if self._is_qwen35() and active_patches:
            if not getattr(self, "_reported_patch_aware_batch", False):
                print(
                    "Qwen3.5 patched candidate scoring: single-pass full-option "
                    "batching enabled"
                )
                self._reported_patch_aware_batch = True

            device = self.kn.model.device
            prompt_tokens = prompt_ids.input_ids[0].tolist()
            if not answer_ids_list or max((len(ids) for ids in answer_ids_list), default=0) == 0:
                return {ans: 1.0 / len(answers) for ans in answers}



            rows = [prompt_tokens + ans_ids[:-1] for ans_ids in answer_ids_list]
            max_len = max(len(row) for row in rows)
            batch_size = len(rows)
            pad_token_id = self.kn.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.kn.tokenizer.eos_token_id or 0
            batch_input_ids = torch.full(
                (batch_size, max_len), pad_token_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (batch_size, max_len), dtype=torch.long, device=device
            )
            scoring_mask = torch.zeros(
                (batch_size, max_len), dtype=torch.bool, device=device
            )
            for row_idx, (row, ans_ids) in enumerate(zip(rows, answer_ids_list)):
                batch_input_ids[row_idx, :len(row)] = torch.as_tensor(
                    row, dtype=torch.long, device=device
                )
                attention_mask[row_idx, :len(row)] = 1
                scoring_mask[
                    row_idx, prompt_len - 1:prompt_len - 1 + len(ans_ids)
                ] = True

            for patch_module in active_patches:
                patch_module.runtime_position_mask = scoring_mask
            try:
                with torch.inference_mode():
                    outputs = self.kn.model(
                        batch_input_ids,
                        attention_mask=attention_mask,
                    )
                    option_log_prob_sums = []
                    option_token_counts = []
                    for row_idx, ans_ids in enumerate(answer_ids_list):
                        score_positions = torch.arange(
                            prompt_len - 1,
                            prompt_len - 1 + len(ans_ids),
                            device=device,
                        )
                        targets = torch.as_tensor(ans_ids, dtype=torch.long, device=device)
                        token_logits = outputs.logits[row_idx, score_positions, :]
                        token_log_probs = torch.log_softmax(token_logits, dim=-1)
                        selected = token_log_probs[
                            torch.arange(len(ans_ids), device=device), targets
                        ]
                        option_log_prob_sums.append(selected.sum().item())
                        option_token_counts.append(len(ans_ids))
            finally:
                for patch_module in active_patches:
                    patch_module.runtime_position_mask = None


            option_log_probs = [
                total / max(count, 1)
                for total, count in zip(option_log_prob_sums, option_token_counts)
            ]
            max_log_prob = max(option_log_probs)
            probs = [math.exp(log_prob - max_log_prob) for log_prob in option_log_probs]
            total = sum(probs)
            if total > 0:
                return {ans: prob / total for ans, prob in zip(answers, probs)}
            return {ans: 1.0 / len(answers) for ans in answers}



        max_opt_len = max(len(ids) for ids in answer_ids_list)
        batch_size = len(answer_ids_list)


        device = self.kn.model.device
        batch_input_ids = torch.full((batch_size, prompt_len + max_opt_len),
                                     self.kn.tokenizer.pad_token_id or 0, dtype=torch.long).to(device)
        attention_mask = torch.zeros((batch_size, prompt_len + max_opt_len), dtype=torch.long).to(device)

        for i, ans_ids in enumerate(answer_ids_list):

            batch_input_ids[i, :prompt_len] = prompt_ids.input_ids[0]

            batch_input_ids[i, prompt_len:prompt_len + len(ans_ids)] = torch.tensor(ans_ids, device=device)

            attention_mask[i, :prompt_len + len(ans_ids)] = 1



        with torch.inference_mode():
            outputs = self.kn.model(batch_input_ids, attention_mask=attention_mask)



        all_log_probs = []
        for i, ans_ids in enumerate(answer_ids_list):
            ans_len = len(ans_ids)
            if self._is_qwen35() or correct_causal_alignment:


                ans_logits = outputs.logits[i, prompt_len - 1:prompt_len + ans_len - 1]
            else:

                ans_logits = outputs.logits[i, prompt_len:prompt_len + ans_len]
            log_probs = torch.log_softmax(ans_logits, dim=-1)


            ans_ids_tensor = torch.tensor(ans_ids, device=device)
            token_indices = torch.arange(ans_len, device=device)
            token_log_probs = log_probs[token_indices, ans_ids_tensor]
            if self._is_qwen35():
                option_log_prob = token_log_probs.mean().item()
            else:
                option_log_prob = token_log_probs.sum().item()
            all_log_probs.append(option_log_prob)


        probs = [math.exp(lp) for lp in all_log_probs]
        total = sum(probs)
        if total > 0:
            results = {ans: p / total for ans, p in zip(answers, probs)}
        else:

            results = {ans: 1.0 / len(answers) for ans in answers}

        return results

    def getPerplexity(self, prompt, ans):
        m = [
            {
                "role": "system",
                "content": "You are a useful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        self.kn.model.eval()

        def incremental_perplexity(prompt, answer):
            prompt_ids = self.kn.tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
                                                               return_tensors="pt").to(self.kn.model.device)

            answer_ids = self.kn.tokenizer.encode(answer, return_tensors="pt", add_special_tokens=False)[0]

            log_probs_sum = 0.0
            total_tokens = len(answer_ids)

            context_ids = prompt_ids[0]

            for i, token_id in enumerate(answer_ids):

                input_ids = torch.tensor(context_ids.ids).unsqueeze(0)
                input_ids = input_ids.cuda()
                with torch.no_grad():
                    outputs = self.kn.model(input_ids)
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                token_log_prob = next_token_log_probs[token_id].item()
                log_probs_sum += token_log_prob

                context_ids = torch.cat([context_ids, torch.tensor([token_id]).to(self.kn.model.device)], dim=0)

            avg_neg_log_likelihood = -log_probs_sum / total_tokens

            perplexity = math.exp(avg_neg_log_likelihood)
            return perplexity, total_tokens

        perplexity, total_tokens = incremental_perplexity(m, ans)

        return perplexity, total_tokens
