import argparse
import json
import random

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.utils import load_json, clean_up


def get_context(idx, dataset_name):
    """
    :return:
        for ECQA: question, choice
        for eSNLI: premise, hypothesis
    """
    if dataset_name == "ECQA":
        df = pd.read_csv("../data/ECQA/ECQA_val.csv")
        return df["texts"][idx], ",".join(df["choices"][idx].split("-"))
    elif dataset_name == "eSNLI":
        df = pd.read_csv("../data/eSNLI/eSNLI_val.csv")
        return df["sentence1"][idx], df["sentence2"][idx]
    elif dataset_name == "healthFC":
        df = pd.read_csv("../data/healthFC/healthFC_val.csv")
        return df["questions"][idx], df["documents"][idx]

      
def load_demonstration(dataset_name, suggestion, num_shot=10):
    prompt_template = ""
    demonstrations = load_json(f"../data/demos/{dataset_name}_demos.json")
    valid_index = []
    for demo in demonstrations:
        if demo["need_improve"]:
            valid_index.append(demo["id"])

    random_list = random.sample(valid_index, num_shot)

    if dataset_name == "ECQA":
        for idx in random_list:
            prompt_template += f"Question: {demonstrations[idx]['question']}\n"
            prompt_template += f"Choices: {demonstrations[idx]['choice']}\n"
            prompt_template += f"Original explanation: {demonstrations[idx]['original_explanation']}\n"
            prompt_template += f"Feedback: {demonstrations[idx]['feedback']}\n"

            if suggestion:
                prompt_template += f"Suggested explanation: {demonstrations[idx]['critic_explanation']}\n\n"
            else:
                prompt_template += "\n"
    elif dataset_name == "eSNLI":
        for idx in random_list:
            prompt_template += f"Premise: {demonstrations[idx]['premise']}\n"
            prompt_template += f"Hypothesis: {demonstrations[idx]['hypothesis']}\n"
            prompt_template += f"Original explanation: {demonstrations[idx]['original_explanation']}\n"
            prompt_template += f"Feedback: {demonstrations[idx]['feedback']}\n"

            if suggestion:
                prompt_template += f"Suggested explanation: {demonstrations[idx]['critic_explanation']}\n\n"
            else:
                prompt_template += "\n"
    elif dataset_name == "healthFC":
        for idx in random_list:
            prompt_template += f"Document: {demonstrations[idx]['document']}\n"
            prompt_template += f"Question: {demonstrations[idx]['question']}\n"
            prompt_template += f"Original explanation: {demonstrations[idx]['original_explanation']}\n"
            prompt_template += f"Feedback: {demonstrations[idx]['feedback']}\n"

            if suggestion:
                prompt_template += f"Suggested explanation: {demonstrations[idx]['critic_explanation']}\n\n"
            else:
                prompt_template += "\n"

    return prompt_template

  
def get_demonstrations_and_template(dataset_name, first_text_field, second_text_field,
                                    initial_explanation, suggestion, feedback=None):
    prompt_template = ""

    if dataset_name == "ECQA":
        if suggestion:
            prompt_template += ("Please review the question, choices, initial explanation, and feedback, then generate "
                                "a revised and improved explanation based on the feedback provided. Below are some "
                                "examples.\n\n")
        else:
            prompt_template += ("Please review the question, choices, and the explanation, and then provide detailed "
                                "feedback on the clarity, accuracy, and comprehensiveness of the explanation. Belows "
                                "are some examples.\n\n")

        prompt_template += load_demonstration(dataset_name, suggestion=suggestion)
        prompt_template += f"Question: {first_text_field}\n"
        prompt_template += f"Choices: {second_text_field}\n"
        prompt_template += f"Original explanation: {initial_explanation}\n"

        if suggestion:
            prompt_template += f"Feedback: {feedback}\n"
            prompt_template += "Suggested explanation: "
        else:
            prompt_template += "Feedback: "

    elif dataset_name == "eSNLI":
        if suggestion:
            prompt_template += ("Please review the premise, hypothesis, initial explanation, and feedback, then generate "
                                "a revised and improved explanation based on the feedback provided. Below are some "
                                "examples.\n\n")
        else:
            prompt_template += ("Please review the premise, hypothesis, and the explanation, and then provide detailed "
                                "feedback on the clarity, accuracy, and comprehensiveness of the explanation. Belows "
                                "are some examples.\n\n")

        prompt_template += load_demonstration(dataset_name, suggestion=suggestion)
        prompt_template += f"Premise: {first_text_field}\n"
        prompt_template += f"Hypothesis: {second_text_field}\n"
        prompt_template += f"Original explanation: {initial_explanation}\n"

        if suggestion:
            prompt_template += f"Feedback: {feedback}\n"
            prompt_template += "Suggested explanation: "
        else:
            prompt_template += "Feedback: "

    elif dataset_name == "healthFC":
            if suggestion:
                prompt_template += (
                    "Please review the document, question, initial explanation, and feedback, then generate "
                    "a revised and improved explanation based on the feedback provided. Below are some "
                    "examples.\n\n")
            else:
                prompt_template += (
                    "Please review the document, question and the explanation, and then provide detailed "
                    "feedback on the clarity, accuracy, and comprehensiveness of the explanation. Below "
                    "are some examples.\n\n")

            prompt_template += load_demonstration(dataset_name, suggestion=suggestion)
            prompt_template += f"Document: {second_text_field}\n"
            prompt_template += f"Question: {first_text_field}\n"
            prompt_template += f"Original explanation: {initial_explanation}\n"

            if suggestion:
                prompt_template += f"Feedback: {feedback}\n"
                prompt_template += "Suggested explanation: "
            else:
                prompt_template += "Feedback: "

    return prompt_template


def feedback_explanation_generation(generator_name, dataset_name, model_name, model, tokenizer, max_new_tokens=64):
    initial_explanation_dict = load_json(
        path=f"../results/explanation_generator/{dataset_name}_generator_explanation_{generator_name}.json")

    initial_explanations = [i["explanation"] for i in initial_explanation_dict]
    feedbacks = []
    suggested_explanations = []

    for idx, _dict in enumerate(initial_explanation_dict):
        initial_explanation = _dict["explanation"]
        first_text_field, second_text_field = get_context(idx, dataset_name)


        prompt_template = get_demonstrations_and_template(dataset_name, first_text_field, second_text_field,
                                                          initial_explanation, suggestion=False,
                                                          feedback="")

        _input = tokenizer(prompt_template, return_tensors="pt")
        input_ids = _input.input_ids.to(device)
        attention_mask = _input.attention_mask.to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                max_new_tokens=max_new_tokens
            )
        decoded = tokenizer.decode(output[0])
        try:
            feedback = decoded.split(prompt_template)[1]
        except:
            feedback = decoded
            if "Feedback: " in feedback:
                feedback = feedback[feedback.rindex("Feedback: "):]
            else:
                print(f"Error! {feedback}")

        feedback = clean_up(feedback)

        feedbacks.append({
            "idx": idx,
            "initial_explanation": initial_explanations[idx],
            "feedback": feedback
        })
        
        print("feedback:", feedback)


    jsonFile = open(
        f"../results/feedback_suggestion/{dataset_name}_generator_{generator_name}_critic_{model_name}_feedback_and_suggestion.json",
        "w")
    jsonString = json.dumps(feedbacks)
    jsonFile.write(jsonString)
    jsonFile.close()

    for idx, _dict in enumerate(initial_explanation_dict):
        initial_explanation = _dict["explanation"]
        first_text_field, second_text_field = get_context(idx, dataset_name)

        prompt_template = get_demonstrations_and_template(dataset_name, generator_name, first_text_field,
                                                          second_text_field, initial_explanation, suggestion=True,
                                                          feedback=feedbacks[idx])

        _input = tokenizer(prompt_template, return_tensors="pt")
        input_ids = _input.input_ids.to(device)
        attention_mask = _input.attention_mask.to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                max_new_tokens=max_new_tokens
            )
        decoded = tokenizer.decode(output[0])
        try:
            suggestion = decoded.split(prompt_template)[1]
        except:
            suggestion = decoded
            if "Suggested explanation: " in suggestion:
                suggestion = suggestion[suggestion.rindex("Suggested explanation: "):]
            else:
                print(f"Error! {suggestion}")

        suggestion = clean_up(suggestion)
        suggested_explanations.append(suggestion)
        print("suggestion:", suggestion)

    content_ls = load_json(f"../results/feedback_suggestion/{dataset_name}_generator_{generator_name}_critic_{model_name}_feedback_and_suggestion.json")
    for i in range(len(content_ls)):
        content_ls[i]["suggestion"] = suggested_explanations[i]

    jsonFile = open(
        f"../results/feedback_suggestion/{dataset_name}_generator_{generator_name}_critic_{model_name}_feedback_and_suggestion.json",
        "w")
    jsonString = json.dumps(content_ls)
    jsonFile.write(jsonString)
    jsonFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator",
        default="Qwen/Qwen2-7B",
        help="Identify which LLM to be the generator",
    )

    parser.add_argument(
        "--critic",
        default="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ",
        help="Identify which LLM to be the critic",
    )

    parser.add_argument(
        "--dataset",
        default="ECQA",
        help="Identify which dataset to use",
    )
    args = parser.parse_args()

    model_name = args.critic
    generator_name = args.generator
    dataset = args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if dataset == "healthFC":
        max_new_tokens = 100
    else:
        max_new_tokens = 64
    feedback_explanation_generation(generator_name.split("/")[1], dataset, model_name.split("/")[1], model=model,
                                    tokenizer=tokenizer, max_new_tokens=max_new_tokens)

