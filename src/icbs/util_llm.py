# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


def buffer_input(sample, max_seqlen, debug=False):
    """Fills in missing values in sample, or truncate, as needed."""
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    n_records, n_tokens = input_ids.shape
    if n_tokens > max_seqlen:
        # Truncate down to max_seqlen
        input_ids = input_ids[:, :max_seqlen]
        attention_mask = attention_mask[:, :max_seqlen]
        n_tokens = max_seqlen
    else:
        # Fill with zeros up to max_seqlen
        n_fill = max_seqlen - n_tokens
        filler = torch.zeros(size=(n_records, n_fill), dtype=torch.int64)
        input_ids = torch.cat((input_ids, filler), dim=1)
        attention_mask = torch.cat((attention_mask, filler), dim=1)

    return {"input_ids": input_ids, "attention_mask": attention_mask}, n_tokens


def prep_loader_batch(batch, tokenizer, seqlen):
    """
    Converts batch coming from dataloader, which contains raw data fields "text",
    "timestamp", "url" into form expected by our pruner. Specifically, this function
    extracts the set of lines in "text", tokenizes the text (converts each string to the
    appropriate token ID via the tokenizer), passes the tokens to buffer_input(),
    collects results, and returns the full collected set.
    """
    # For the non-LLM case, it's convenient to do nothing - just return the batch
    # as is
    if tokenizer is None:
        return batch

    content = batch["text"]
    in_cache = []
    target_cache = []
    for line in content:
        line_tokens = tokenizer(line, return_tensors="pt")
        in_buffered, n_tokens = buffer_input(line_tokens, seqlen)
        in_cache.append(in_buffered)

        targets = in_buffered["input_ids"].clone()
        # Anywhere filler is present in inputs, mask output with -100
        targets[:, n_tokens:] = -100
        target_cache.append(targets)

    return in_cache, target_cache


def is_llm(model):
    """Returns True if the model is an LLM, False otherwise."""
    return isinstance(
        model,
        (
            AutoModelForCausalLM,
            GPTNeoForCausalLM,
            GPTNeoXForCausalLM,
            LlamaForCausalLM,
            MistralForCausalLM,
        ),
    ) or hasattr(model, "lm_head")


def load_llm(
    model_name,
    device,
    cache_folder="llm_cache",
    offload_folder="llm_offload",
    max_seqlen=4096,
):
    """Loads an LLM."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # Keep memory reduced (vs. float32) - float16 only works on GPU!
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        cache_dir=cache_folder,
        low_cpu_mem_usage=True,
        device_map="cpu" if device == "cpu" else "auto",
        offload_folder=offload_folder,
    )
    model.seqlen = min(model.config.max_position_embeddings, max_seqlen)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


def prepare_llm(model, device):
    """Prepares the LLM for use with the pruning loop."""
    try:
        model, tokenizer = model
        seqlen = model.seqlen

        if len(model.hf_device_map) > 1:
            input_device = model.hf_device_map["model.embed_tokens"]
        else:
            # If everything fits on one device, the device map will fallback to a single
            # entry (empty string) mapping to the device
            input_device = model.hf_device_map[""]
            print(f"Device map: {model.hf_device_map}")

    except (TypeError, AttributeError):
        tokenizer = None
        seqlen = None
        input_device = device

    return model, tokenizer, seqlen, input_device
