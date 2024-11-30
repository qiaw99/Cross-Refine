import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.critic_feedback_explanation_generation import get_context
from experiments.feedback_refinement import get_prompt_template
from experiments.generator_explanation import prompt_template_model_generation
from experiments.utils import load_json


def select_examples(feedback_and_suggestion_ls, num_example=30):
    """
    select instances with available and meaningful feedback and suggestions
    :return: randomly selected ids
    """
    idx_ls = []

    for i in feedback_and_suggestion_ls:
        if i["feedback"] != "" and i["suggested_explanation"] != "" and not i["suggested_explanation"].startswith("1"):
            idx_ls.append(i["idx"])

    res = random.sample(idx_ls, num_example)
    res.sort()

    return res


def ablation_study(dataset_name, critic_name, model_name, tokenizer, model):
    """
    Ablation study: generate explanation with/without certain components (feedback & suggestion)
    """
    feedback_and_suggestion_ls = load_json(f"../results/feedback_suggestion/{dataset_name}/{dataset_name}_generator_{model_name}_critic_{critic_name}_feedback_and_suggestion.json")
    initial_explanation_ls = load_json(f"../results/explanation_generator/{dataset_name}_generator_explanation_{model_name}.json")
    idx_ls = select_examples(feedback_and_suggestion_ls)

    combinations = [
        {
            "feedback": False,
            "suggestion": True,
            "prompt": "You are an excellent assistant to improve explanations through suggested explanation. Your task is to "
                      "refine an initial explanation based on the suggested explanation provided. Belows are some examples.\n\n"
        },
        {
            "feedback": True,
            "suggestion": False,
            "prompt": "You are an excellent assistant to improve explanations through feedback. Your task is to "
                      "refine an initial explanation based on the feedback provided. Belows are some examples.\n\n"

        }
    ]

    for combi in combinations:
        res = []
        for idx in idx_ls:
            if combi["feedback"]:
                feedback = feedback_and_suggestion_ls[idx]["feedback"]
            else:
                feedback = None

            if combi["suggestion"]:
                suggestion = feedback_and_suggestion_ls[idx]["suggested_explanation"]
            else:
                suggestion = None

            prompt_template = ""
            first_text_field, second_text_field = get_context(idx, dataset_name)
            prompt_template += get_prompt_template(dataset_name, first_text_field, second_text_field, initial_explanation_ls[idx]["explanation"], suggestion, feedback, combi["prompt"])
            # print(prompt_template)

            explanation = prompt_template_model_generation(model, tokenizer, prompt_template, max_new_tokens=128)

            print({
                "idx": idx,
                "refined_explanation": explanation
            })

            res.append({
                "idx": idx,
                "refined_explanation": explanation
            })

            jsonFile = open(
                f"../results/refined_explanation/{dataset_name}_generator_{generator_name}_critic_{critic_name}_feedback_{combi['feedback']}_suggestion_{combi['suggestion']}_ablation_study.json",
                "w")
            jsonString = json.dumps(res)
            jsonFile.write(jsonString)
            jsonFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    critic_name = "TechxGenus/Meta-Llama-3-70B-GPTQ"
    generator_name = "Qwen/Qwen2-7B"

    parser.add_argument(
        "--dataset",
        default="ECQA",
        help="Identify which dataset to use",
    )

    args = parser.parse_args()
    dataset = args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(generator_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    ablation_study(dataset, critic_name, generator_name.split("/")[1], tokenizer, model)
    # ablation_study("ECQA", "Qwen2-7B", "Meta-Llama-3-70B-GPTQ", None, None)