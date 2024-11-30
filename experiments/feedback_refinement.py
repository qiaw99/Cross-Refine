import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_json, clean_up
from experiments.critic_feedback_explanation_generation import get_context


def get_demonstrations(dataset_name, suggestion, feedback, num_shot=10):
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
            prompt_template += f"Initial explanation: {demonstrations[idx]['original_explanation']}\n"

            if feedback is not None:
                prompt_template += f"Feedback: {demonstrations[idx]['feedback']}\n"

            if suggestion is not None:
                prompt_template += f"Suggestion: {demonstrations[idx]['critic_explanation']}\n"

            prompt_template += f"Refined explanation: {demonstrations[idx]['refined_explanation']}\n\n"
    elif dataset_name == "eSNLI":
        for idx in random_list:
            prompt_template += f"Premise: {demonstrations[idx]['premise']}\n"
            prompt_template += f"Hypothesis: {demonstrations[idx]['hypothesis']}\n"
            prompt_template += f"Initial explanation: {demonstrations[idx]['original_explanation']}\n"
            
            if feedback is not None:
                prompt_template += f"Feedback: {demonstrations[idx]['feedback']}\n"

            if suggestion is not None:
                prompt_template += f"Suggestion: {demonstrations[idx]['critic_explanation']}\n"
            prompt_template += f"Refined explanation: {demonstrations[idx]['refined_explanation']}\n\n"
    elif dataset_name == "healthFC":
        for idx in random_list:
            prompt_template += f"Document: {demonstrations[idx]['document']}\n"
            prompt_template += f"Question: {demonstrations[idx]['question']}\n"
            prompt_template += f"Initial explanation: {demonstrations[idx]['original_explanation']}\n"
            
            if feedback is not None:
                prompt_template += f"Feedback: {demonstrations[idx]['feedback']}\n"

            if suggestion is not None:
                prompt_template += f"Suggestion: {demonstrations[idx]['critic_explanation']}\n"
            prompt_template += f"Refined explanation: {demonstrations[idx]['refined_explanation']}\n\n"

    return prompt_template


def get_prompt_template(dataset_name, first_text_field, second_text_field,
                        initial_explanation, suggestion, feedback, prompt_template=None):
    if prompt_template is None:
        prompt_template = ("You are an excellent assistant to improve explanations through feedback. "
                           "Your task is to refine an initial explanation based on the feedback provided "
                           "and suggest a more accurate and clear explanation. Belows are some examples.\n\n")
    prompt_template += get_demonstrations(dataset_name, suggestion, feedback)

    if dataset_name == "ECQA":
        prompt_template += f"Question: {first_text_field}\n"
        prompt_template += f"Choices: {second_text_field}\n"
    elif dataset_name == "eSNLI":
        prompt_template += f"Premise: {first_text_field}\n"
        prompt_template += f"Hypothesis: {second_text_field}\n"
    elif dataset_name == "HealthFC":
        prompt_template += f"Document: {second_text_field}\n"
        prompt_template += f"Question: {first_text_field}\n"

    prompt_template += f"Initial explanation: {initial_explanation}\n"

    if feedback is not None:
        prompt_template += f"Feedback: {feedback}\n"

    if suggestion is not None:
        prompt_template += f"Suggestion: {suggestion}\n"

    prompt_template += f"Refined explanation: "

    return prompt_template


def explanation_refinement(generator_name, dataset_name, critic_name, model, tokenizer):
    initial_explanation_ls = load_json(
        f"../results/explanation_generator/{dataset_name}_generator_explanation_{generator_name}.json")
    feedback_suggestion_ls = load_json(
        f"../results/feedback_suggestion/{dataset_name}_generator_{generator_name}_critic_{critic_name}_feedback_and_suggestion.json")

    refined_explanations = []

    minimum = min(len(initial_explanation_ls), len(feedback_suggestion_ls))

    for idx in range(minimum):
        if initial_explanation_ls[idx]["explanation"] != "" and feedback_suggestion_ls[idx]["suggestion"] != "":
            prompt_template = ""
            first_text_field, second_text_field = get_context(idx, dataset_name)
            prompt_template += get_prompt_template(dataset_name, first_text_field, second_text_field,
                                                   initial_explanation_ls[idx]["explanation"],
                                                   feedback_suggestion_ls[idx]["suggestion"],
                                                   feedback_suggestion_ls[idx]["feedback"])

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
                    max_new_tokens=128
                )

            decoded = tokenizer.decode(output[0])
            try:
                refined_explanation = decoded.split(prompt_template)[1]
            except:
                refined_explanation = decoded
                if "Refined explanation: " in refined_explanation:
                    refined_explanation = refined_explanation[refined_explanation.rindex("Refined explanation: "):]
                else:
                    print(f"Error! {refined_explanation}")

                refined_explanation = f"Error! {tokenizer.decode(output[0]).split(prompt_template)}"
        
            refined_explanation = clean_up(refined_explanation)

            refined_explanations.append({
                "idx": idx,
                "refined_explanation": refined_explanation,
            })

    jsonFile = open(
        f"../results/refined_explanation/{dataset_name}_generator_{generator_name}_critic_{critic_name}_refined_explanation.json",
        "w")
    jsonString = json.dumps(refined_explanations)
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

    # load the generator
    model = AutoModelForCausalLM.from_pretrained(generator_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    explanation_refinement(generator_name.split("/")[1], dataset, model_name.split("/")[1], model=model,
                           tokenizer=tokenizer)
    # explanation_refinement("Qwen2-7B", dataset, "Meta-Llama-3-8B-Instruct-GPTQ", model=None,
    #                        tokenizer=None)
