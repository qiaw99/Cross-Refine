import argparse
import json
import random

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.guided_decoding.gd_logit_processor import GuidedParser, GuidedDecodingLogitsProcessor
from experiments.guided_decoding.grammar import improvement_grammar
from experiments.utils import load_json


def load_demonstrations(path, dataset_name, first_field, second_field, explanation, num_shot=3):
    """loading #num_shot demonstrations checking if explanations need to be improved"""
    demonstrations = load_json(path)
    random_idx_list = [random.randint(0, len(demonstrations) - 1) for i in range(num_shot)]
    prompt_template = ""

    if dataset_name == "ECQA":
        for idx in random_idx_list:
            prompt_template += f"Question: {demonstrations[idx]['question']}\n"
            prompt_template += f"Choices: {demonstrations[idx]['choice']}\n"
            prompt_template += f"Explanation: {demonstrations[idx]['original_explanation']}\n"
            prompt_template += f"Improvement needed: {demonstrations[idx]['need_improve']}\n\n"

        prompt_template += f"Question: {first_field}\n"
        prompt_template += f"Choices: {second_field}\n"
        prompt_template += f"Explanation: {explanation}\n"
        prompt_template += f"Improvement needed: "
    elif dataset_name == "eSNLI":
        for idx in random_idx_list:
            prompt_template += f"Premise: {demonstrations[idx]['premise']}\n"
            prompt_template += f"Hypothesis: {demonstrations[idx]['hypothesis']}\n"
            prompt_template += f"Explanation: {demonstrations[idx]['original_explanation']}\n"
            prompt_template += f"Improvement needed: {demonstrations[idx]['need_improve']}\n\n"

        prompt_template += f"Premise: {first_field}\n"
        prompt_template += f"Premise: {second_field}\n"
        prompt_template += f"Explanation: {explanation}\n"
        prompt_template += f"Improvement needed: "
    elif dataset_name == "healthFC":
        for idx in random_idx_list:
            prompt_template += f"Question: {demonstrations[idx]['question']}\n"
            prompt_template += f"Document: {demonstrations[idx]['document']}\n"
            prompt_template += f"Explanation: {demonstrations[idx]['original_explanation']}\n"
            prompt_template += f"Improvement needed: {demonstrations[idx]['need_improve']}\n\n"

        prompt_template += f"Question: {first_field}\n"
        prompt_template += f"Document: {second_field}\n"
        prompt_template += f"Explanation: {explanation}\n"
        prompt_template += f"Improvement needed: "

    return prompt_template


def check_improvements_needed(first_field, second_field, explanation, dataset_name, tokenizer, model, device="cuda"):
    if dataset_name == "ECQA":
        prompt_template = ("You are tasked with evaluating explanations for multiple-choice questions. Each "
                           "explanation needs to be clear, accurate, and helpful for understanding the correct "
                           "answer. Given a question, its choices, and a provided explanation, determine if the "
                           "explanation needs improvement. Below are some examples:\n")

    elif dataset_name == "eSNLI":
        prompt_template = ("You are tasked with evaluating explanations for natural language inference tasks. Each "
                           "explanation needs to be clear, accurate, and helpful for understanding the relationship "
                           "between a premise and a hypothesis. Given a premise, a hypothesis, and a provided "
                           "explanation, determine if the explanation needs improvement.")
    elif dataset_name == "HealthFC":
        prompt_template = ("You are tasked with evaluating explanations for fact-checking questions given a document which might contain some relevant evidence. Each "
                           "explanation needs to be clear, accurate, and helpful for understanding the correct "
                           "answer. Given a question, the corresponding document, and a provided explanation, determine if the "
                           "explanation needs improvement. Below are some examples:\n")

    prompt_template += load_demonstrations(f"../data/demos/{dataset_name}_demos.json", dataset_name, first_field, second_field, explanation)
    # print(prompt_template)

    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(device)

    parser = GuidedParser(improvement_grammar, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
    guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

    with torch.no_grad():
        output = model.greedy_search(
            input_ids=input_ids,
            logits_processor=guided_preprocessor,
            eos_token_id=parser.eos_token,
            pad_token_id=model.config.pad_token_id,
        )
    try:
        result = tokenizer.decode(output[0]).split(prompt_template)[1].split(" [e]")[0].strip()
        print(f"prediction: {result}")
    except:
        result = tokenizer.decode(output[0]).split(prompt_template)
        print(f"Error! prediction: {result}")

    return result


def explanation_evaluation(dataset_name, model_name, tokenizer, model):
    content_ls = load_json(f"../results/explanation_generator/{dataset_name}_generator_explanation_{model_name}_CoT.json")
    df = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_val.csv")
    predictions = []

    if dataset_name == "ECQA":
        first_fields = list(df["texts"])
        second_fields = list(df["choices"])

        for i in range(len(second_fields)):
            # print(second_fields[i])
            second_fields[i] = ",".join(second_fields[i].split("-"))
    elif dataset_name == "eSNLI":
        first_fields = list(df["sentence1"])[:1000]
        second_fields = list(df["sentence2"])[:1000]
    elif dataset_name == "healthFC":
        first_fields = list(df["questions"])
        second_fields = list(df["documents"])

    for idx, i, first_field, second_field in zip([i for i in range(len(content_ls))], content_ls, first_fields, second_fields):
        explanation = i["explanation"]
        predictions.append({
            "idx": idx,
            "need_improve": check_improvements_needed(first_field, second_field, explanation, dataset_name, tokenizer, model)
        })

    jsonFile = open(f"../results/prediction/{dataset_name}_improvement_prediction_{model_name}.json", "w")
    jsonString = json.dumps(predictions)
    jsonFile.write(jsonString)
    jsonFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="TechxGenus/Meta-Llama-3-70B-GPTQ",
        help="Identify which LLM to use",
    )

    parser.add_argument(
        "--dataset",
        default="ECQA",
        help="Identify which dataset to use",
    )
    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    explanation_evaluation(dataset, model_name.split("/")[1], tokenizer, model)
    # explanation_evaluation("ECQA", "Qwen2-7B", tokenizer, model)
