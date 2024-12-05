"""
This sample, non-production-ready code includes evaluation functions.

© 2024 Amazon Web Services, Inc. or its affiliates. All Rights Reserved. This AWS
Content is provided subject to the terms of the AWS Customer Agreement available at
http://aws.amazon.com/agreement or other written agreement between Customer and either
Amazon Web Services, Inc. or Amazon Web Services EMEA SARL or both.
"""

import os
import time

# TODO - unable to handle this without an os.environ override..
#  lm_eval claims to have fixed this, but it doesn't seem to be reading the
#  trust_remote_code param from model_args. Might have old library version? This Issue
#  is about invoking via CLI, but covers the problem: https://github.com/EleutherAI/lm-evaluation-harness/pull/1998
# 2024-08-20:00:40:00,124 WARNING  [task.py:815] [Task: boolq] metric acc is defined, but higher_is_better is not. using default higher_is_better=True
# The repository for super_glue contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/super_glue.
# You can avoid this prompt in future by passing the argument `trust_remote_code=True`.
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "True"

import lm_eval
import torch
from icbs.loops import valid_loop, valid_loop_llm
from icbs.util import clear_memory_caches
from icbs.util_llm import is_llm
from lm_eval.models.huggingface import HFLM


def evaluate_model(dataloader, model, loss_function, device, verbose=True):
    """Evaluates the model on data from the dataloader.

    Note: the metric used is perplexity for LLMs and accuracy for all other models.
    """
    # Choose the right validation loop for the model type
    valid_loop_ = valid_loop_llm if is_llm(model) else valid_loop

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    loss, metric = valid_loop_(dataloader, model, loss_function, device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    evaluation_time = time.perf_counter() - start_time

    if verbose:
        print(f"  Evaluation time: {evaluation_time=:.1f}sec")

    return metric, loss, evaluation_time


def evaluate_model_train_and_valid(
    train_dataloader, valid_dataloader, model, loss_function, device, verbose=True
):
    """Evaluates the model on data from the train and valid dataloaders."""
    if verbose:
        print("Train:")
    train_accuracy, train_loss, train_evaluation_time = evaluate_model(
        train_dataloader, model, loss_function, device, verbose
    )

    if verbose:
        print("Valid:")
    valid_accuracy, valid_loss, valid_evaluation_time = evaluate_model(
        valid_dataloader, model, loss_function, device, verbose
    )

    evaluation_time = train_evaluation_time + valid_evaluation_time

    return (
        train_accuracy,
        train_loss,
        valid_accuracy,
        valid_loss,
        evaluation_time,
    )


def evaluate_model_tasks(
    tasks,
    model,
    tokenizer,
    verbose=True,
    **kwargs,
):
    """Evaluates the model on tasks in evaluation harness.

    Note: Returns the results dict as returned by simple_evaluate(). Additional keyword
    arguments are passed to simple_evaluate(), see here for documentation:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    and here for the code and additional documentation:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py
    """
    if verbose:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

    # In order to pass an already instantiated model to simple_evaluate(), we need to
    # convert it to an LM object, and we use HFLM since our models are from Hugging
    # Face. For more info, see here:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/models/huggingface.py
    model = OurHFLM(pretrained=model, tokenizer=tokenizer)
    results = lm_eval.evaluator.simple_evaluate(
        model=model,
        model_args="",  # Ignored anyway if model is a str, as we do here
        tasks=tasks,
        verbosity="DEBUG" if verbose else "INFO",
        **kwargs,
    )
    clear_memory_caches()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    evaluation_time = time.perf_counter() - start_time

    if verbose:
        print("Results for each task:")

    average_accuracy = 0
    for task, task_results in results["results"].items():
        accuracy = task_results["acc,none"] * 100
        error = task_results["acc_stderr,none"] * 100
        average_accuracy += accuracy

        if verbose:
            print(f"    {task} : {accuracy:.3f} (+-{error:.3f})")

    average_accuracy /= len(results["results"])

    if verbose:
        print(
            f"  Evaluation time: {evaluation_time=:.1f}sec, Accuracy: {average_accuracy:.2f}% (+-{error:.2f}%)"
        )

    return average_accuracy, evaluation_time, results


class OurHFLM(HFLM):
    """Identical to HFLM - gets rid of some unnecessary warnings."""

    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.

        Same as the parent's method with the same name, but removing all the SHA stuff,
        since it results in a warning that prints the model to the log every time.
        This is a small bug in lm-evaluation-harness, for the case in which pretrained
        is not a string (i.e., we are using a custom model).
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
        }
        return model_info
