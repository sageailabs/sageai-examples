import json
import re
from json.decoder import JSONDecodeError
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from termcolor import colored
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


INSTRUCTION_TEMPLATE_ZOO: Dict[str, str] = {
    "straightforward": """Given this question `{question}` tell me if this is asking about accounting terminology. Respond with a 'yes' or a 'no' and then explain your reasoning. Use this JSON format for your response {{"answer": "yes/no", "reasoning": <REASONING>}} Use only valid JSON in your response.""",
    "just_answer": """Given this question `{question}` tell me if this is asking about accounting terminology. Respond with a 'yes' or a 'no' answer. Use this JSON format for your response {{"answer": "yes/no"}} Use only valid JSON in your response.""",
    "accounting_related": """Given this question `{question}` tell me if this is asking about something accounting related. Respond with a 'yes' or a 'no' and then explain your reasoning. Use this JSON format for your response {{"answer": "yes/no", "reasoning": <REASONING>}} Use only valid JSON in your response.""",
}


def answer_to_bool(answer: str) -> bool:
    """
    >>> answer_to_bool(answer="yes")

    Converts an answer string from a model to a boolean value.
    This is necessary for future automated analysis and computing statistics.

    :param answer: answer string from model
    :return: bool representation of answer
    """
    return answer.lower().strip() in ["yes", "true", "y"]


def parse_json_from_response(response: str) -> Dict[str, str]:
    """
    We're instructing our model to yield json formatted responses.
    This function extracts that json from the response and returns it as a dictionary.

    :param response: response from model
    :return: json-formatted dictionary of model's response
    """
    answers_string = re.findall("({[^}]*})", response)[-1]
    try:
        json_from_string = json.loads(answers_string)
        return json_from_string
    except JSONDecodeError:
        print(answers_string)


def format_prompt_for_evaluation(instruction_template: str, question: str) -> str:
    """
    Fetches a prompt format from INSTRUCTION_TEMPLATE_ZOO based in the `instruction_template` selected
    and formats the template with a given `question`.

    :param instruction_template: instruction template to select from INSTRUCTION_TEMPLATE_ZOO
    :param question: question to send to model
    :return: Mistral-formatted instruction string
    """
    template = INSTRUCTION_TEMPLATE_ZOO[instruction_template]
    formatted_prompt = template.format(question=question)
    return f"""[INST] {formatted_prompt} [/INST]"""


def custom_template(
    model: MistralForCausalLM,
    tokenizer: LlamaTokenizerFast,
    prompt: str,
    max_length: int = 200,
) -> str:
    """
    This function sends our formatted prompts (formatted by the format_prompt_for_evaluation function) to the
    model, decodes the model's response sequence of tokens, and returns the decoded response string.

    :param model: Mistral model object
    :param tokenizer: tokenizer model object
    :param prompt: Mistral-compliant formatted prompt
    :param max_length: max length of generated text
    :return:
    """
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_ids = model_inputs["input_ids"][:, 0:-1]
    attention_mask = model_inputs["attention_mask"][:, 0:-1]
    result = model.generate(
        input_ids,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        output_attentions=True,
        output_hidden_states=True,
        max_new_tokens=max_length,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )
    x = result.sequences
    decoded = tokenizer.batch_decode(x)
    return decoded[0]


def collect_responses(
    dataset: pd.DataFrame,
    model: MistralForCausalLM,
    instruction_template: str,
    tokenizer: LlamaTokenizerFast,
    debug: bool = False,
) -> None:
    """
    This function handles iterating over the given dataset, collecting responses (returned by the
    custom_template function) and adding those responses to the dataset for analysis in the calculate_stats function.

    :param dataset: dataframe containing dataset of test questions
    :param model: Mistral model object
    :param instruction_template: instruction template to select from INSTRUCTION_TEMPLATE_ZOO
    :param tokenizer: tokenizer
    :param debug: debug flag for printing invalid responses
    :return:
    """
    model_responses = []
    model_response_bools = []
    num_skipped = 0
    responses_with_invalid_formats = []
    for idx, row in dataset.iterrows():
        prompt = format_prompt_for_evaluation(
            instruction_template=instruction_template, question=row["question"]
        )
        response = custom_template(model, tokenizer=tokenizer, prompt=prompt)
        if debug:
            print(response)
        try:
            response_json = parse_json_from_response(response=response)
            answer_bool = answer_to_bool(answer=response_json["answer"])
            model_responses.append(response_json)
            model_response_bools.append(answer_bool)
        except JSONDecodeError as jde:
            if debug:
                print(jde)
            # Drop the row from our resulting dataset to avoid polluting summary stats.
            num_skipped += 1
            dataset.drop(idx, axis=0, inplace=True)
            # Collect the invalid response for future analysis.
            responses_with_invalid_formats.append(response)
    print(colored(prompt, "red"))
    print(colored(response, "green"))

    dataset["response"] = model_responses
    dataset["response_bool"] = model_response_bools
    if debug:
        print(colored(f"Number of rows skipped due to formatting issues: {num_skipped}", "red"))
        print(colored("Invalid responses:", "red"))
        print(colored(responses_with_invalid_formats, "red"))


def calculate_stats(dataset: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary statistics based on the responses captured in the dataset by the collect_responses function.

    :param dataset: dataframe containing dataset of test questions and model responses collected by the
    collect_responses function
    :return: dictionary of summary statistics calculated on model's responses
    """
    labels = dataset["label"]
    response_bools = dataset["response_bool"]
    accuracy = accuracy_score(y_true=labels, y_pred=response_bools)
    cm = confusion_matrix(y_true=labels, y_pred=response_bools)
    precision = precision_score(y_true=labels, y_pred=response_bools)
    recall = recall_score(y_true=labels, y_pred=response_bools)
    f1 = f1_score(y_true=labels, y_pred=response_bools)

    # Used for calculating specificity.
    true_negatives = np.logical_and(np.logical_not(response_bools), np.logical_not(labels))
    false_positives = np.logical_and(response_bools, np.logical_not(labels))
    num_true_negatives = np.count_nonzero(true_negatives)
    num_false_positives = np.count_nonzero(false_positives)
    specificity = num_true_negatives / (num_true_negatives + num_false_positives)

    return {
        "accuracy": accuracy,
        "specificity": specificity,
        "precision": precision,
        "confusion_matrix": cm,
        "recall": recall,
        "f1_score": f1,
    }


def display_stats(stats_dict: Dict[str, Any]) -> None:
    """
    Displays summary statistics generated by the calculate_stats function and plots the confusion matrix
    for the evaluation run.

    :param stats_dict: dictionary of summary statistics generated by the calculate_stats function
    :return: None
    """
    display_cm = ConfusionMatrixDisplay(
        confusion_matrix=stats_dict["confusion_matrix"],
        display_labels=["negatives", "positives"],
    )
    print(colored(stats_dict, "magenta"))
    display_cm.plot()
    plt.show()
