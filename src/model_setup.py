"""Bulid the model and prepring the prompts."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable
from typing import cast

from llm_sdk import Small_LLM_Model

from src.function_call_builder import generate_function_call
from src.schemas import FunctionCall, FunctionDefinition, ParameterDefinition
from src.vocabulary import load_vocabulary


def create_model(model_name: str) -> tuple[object, dict[int, str]]:
    model = Small_LLM_Model(model_name=model_name)
    vocabulary = load_vocabulary(_vocabulary_path(model))
    return model, vocabulary


def choose_function_call(
    model: object,
    vocabulary: dict[int, str],
    prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCall:
    context_prompt = _selection_prompt(prompt, functions)
    return generate_function_call(
        model, vocabulary, context_prompt, prompt, functions, _argument_prompt
    )


def _selection_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the natural-language context the model can understand it.

    Only names and descriptions are listed: they are what decides which
    function fits, and every generation step re-reads this whole context,
    so leaving the parameter schemas out keeps each step cheap.
    """
    catalogue = "\n".join(
        f"- {function.name}: {function.description}" for function in functions
    )
    return (
        "Select exactly one function for the user prompt. "
        "Do not answer the prompt.\n"
        f"Functions:\n{catalogue}\nPrompt: {prompt}\nCall: "
    )


def _argument_prompt(prompt: str, function: FunctionDefinition) -> str:
    """Build the context used while the argument values are generated.

    Once the name is chosen the other functions are noise, so only the
    signature and description of the chosen one are kept.
    """
    signature = ", ".join(
        f"{name}: {_type_hint(definition)}"
        for name, definition in function.parameters.items()
    )
    return (
        "Extract the arguments of the function from the prompt.\n"
        f"{function.name}({signature}): {function.description}\n"
        f"Prompt: {prompt}\nCall: "
    )


def _type_hint(definition: ParameterDefinition) -> str:
    """Describe one parameter type in a few words."""
    if definition.enum:
        return "one of " + json.dumps(definition.enum, ensure_ascii=False)
    return definition.type


def _vocabulary_path(model: object) -> Path:
    """a vocabulary path can be diffrennt  so
    Return a public SDK vocabulary path, accepting SDK version differences."""
    for method_name in (
        "get_path_to_vocab_file",
        "get_path_to_vocabulary_json",
        "get_path_to_tokenizer_file",
    ):
        getter = getattr(model, method_name, None)
        if callable(getter):
            typed_getter = cast(Callable[[], str], getter)
            return Path(typed_getter())
    raise TypeError("model does not provide a public vocabulary path method")
