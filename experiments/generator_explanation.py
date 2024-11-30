import argparse
import json
import os
import random
import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from prediction import load_dataset
from utils import clean_up

def save_explanation(explanation_path, _id, explanation):
    """save generated explanation in real time to json file"""
    path = f"results/explanation_generator/{explanation_path}.json"
    if os.path.isfile(path):
        jsonFile = open(path, "r")

        jsonContent = jsonFile.read()
        content_ls = json.loads(jsonContent)
        jsonFile.close()

        content_ls.append({
            "id": _id,
            "explanation": explanation
        })

        jsonString = json.dumps(content_ls)
        jsonFile = open(path, "w")
        jsonFile.write(jsonString)
        jsonFile.close()
    else:
        jsonFile = open(path, "w")
        content_ls = [{
            "id": _id,
            "explanation": explanation
        }]
        jsonString = json.dumps(content_ls)
        jsonFile.write(jsonString)
        jsonFile.close()


def load_prediction(_id, dataset_name, model_name):
    """load predictions according to specific dataset and LLM"""
    if dataset_name == "healthFC":
            fileObject = open(f"results/prediction/{dataset_name}_prediction_{model_name}.json")
    else:
        fileObject = open(f"results/prediction/{dataset_name}_prediction_{model_name}_20shots.json")
    jsonContent = fileObject.read()
    prediction_ls = json.loads(jsonContent)

    return prediction_ls[_id]["prediction"]


def ecqa_template(_id, text, choice, dataset_name, model_name):
    """ECQA task template for NLEs generations"""
    prompt_template = f"Question: {text}\n"

    prompt_template += "Choices: "
    for c in choice.split("-")[:-1]:
        prompt_template += f"{c}, "
    prompt_template += f"{choice.split('-')[-1]}\n"

    prompt_template += f"Prediction: {load_prediction(_id, dataset_name, model_name)}\n"
    prompt_template += f"Explanation:"

    return prompt_template


def esnli_template(_id, s1, s2, dataset_name, model_name):
    """eSNLI task template for NLEs generations"""
    prompt_template = f"Premise: {s1}\n"
    prompt_template += f"Hypothesis: {s2}\n"
    prompt_template += f"Relationship: {load_prediction(_id, dataset_name, model_name)}\n"
    prompt_template += f"Explanation:"

    return prompt_template



def healthFC_template(_id, question, document, dataset_name, model_name):
    """HealthFC task template for NLEs generations"""
    prompt_template = "Now generate explanation given the following document and question, but please do not repeat the document and question text and do not generate any new questions or documents.\n"
    prompt_template += f"Document: {document}\n"
    prompt_template += f"Question: {question}\n"
    prompt_template += f"Prediction: {load_prediction(_id, dataset_name, model_name)}\n"
    prompt_template += f"Explanation:"

    return prompt_template


def get_demonstrations(first_text_field, second_text_field, demo_answers, demo_explanations, num_shot=20,
                       dataset_name="ECQA"):
    """gather demonstrations from train set randomly"""
    random_idx_list = [random.randint(0, len(demo_answers)-1) for _ in range(num_shot)]

    demo_prompt = ""

    if dataset_name == "ECQA":

        demo_prompt += (f"Given a commonsense question, 5 possible answer choices and answer, please provide an "
                        f"explanation justifying why the selected answer choice is the best option based on "
                        f"general world knowledge. ")
        demo_prompt += "Below are some examples.\n"

        for _id in random_idx_list:
            demo_prompt += f"Question: {first_text_field[_id]}\n"
            demo_prompt += "Choices: "
            for c in second_text_field[_id].split("-")[:-1]:
                demo_prompt += f"{c}, "
            demo_prompt += f"{second_text_field[_id].split('-')[-1]}\n"

            demo_prompt += f"Prediction: {second_text_field[_id].split('-')[int(demo_answers[_id])]}\n"
            demo_prompt += f"Explanation: {demo_explanations[_id]}\n\n"

    elif dataset_name == "eSNLI":
        demo_prompt += (f"Given premise, hypothesis and predicted relationship between premise and hypothesis, "
                        f"please provide the reason why the relationship is correct. Below are some examples.\n")
        for _id in random_idx_list:
            demo_prompt += f"Premise: {first_text_field[_id]}\n"
            demo_prompt += f"Hypothesis: {second_text_field[_id]}\n"
            demo_prompt += f"Relationship: {demo_answers[_id]}\n"
            demo_prompt += f"Explanation: {demo_explanations[_id]}\n\n"

    elif dataset_name == "healthFC":
        demo_prompt += f"Given a question, corresponding document text and predicted label: yes, no or unknown, provide a justification for the choice. Below are some examples:\n"
        for _id in random_idx_list:
            demo_prompt += f"Question: {first_text_field[_id]}\n"
            demo_prompt += f"Document: {second_text_field[_id]}\n"
            demo_prompt += f"Answer: {demo_answers[_id]}\n"
            demo_prompt += f"Explanation: {demo_explanations[_id]}\n\n"
    return demo_prompt


def prompt_template_model_generation(model, tokenizer, prompt_template, device="cuda", max_new_tokens=64):
    """LLM generation based on provided prompt template"""
    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            max_new_tokens=max_new_tokens
        )
    decoded = tokenizer.decode(output[0])
    try:
        resulted_explanation = decoded.split(prompt_template)[1]
        print(resulted_explanation)
    except:
        resulted_explanation = decoded
        if "Explanation: " in resulted_explanation:
            resulted_explanation = resulted_explanation[resulted_explanation.rindex("Explanation: "):]
        else:
            print(f"Error! {tokenizer.decode(output[0]).split(prompt_template)}")
    resulted_explanation = clean_up(resulted_explanation)

    return resulted_explanation


def ECQA_generator_explanation(dataset_name, model_name, model, tokenizer):
    first_text_field, second_text_field, demo_answers, demo_explanations = load_dataset("ECQA_train")
    texts, choices, answers, explanations = load_dataset("ECQA_val")

    for idx, text, choice, explanation in zip([i for i in range(len(texts))], texts, choices, explanations):
        prompt_template = get_demonstrations(first_text_field, second_text_field, demo_answers, demo_explanations)
        prompt_template += ecqa_template(idx, text, choice, dataset_name, model_name)

        generator_explanation = prompt_template_model_generation(model, tokenizer, prompt_template)
        save_explanation(f"ECQA_generator_explanation_{model_name}", idx, generator_explanation)


def eSNLI_generator_explanation(dataset_name, model_name, model, tokenizer):
    demo_answers, demo_first_text_field, demo_second_text_field, demo_explanations = load_dataset("eSNLI_train")
    answers, first_text_field, second_text_field, explanations = load_dataset("eSNLI_val")

    for idx, first, second, explanation in zip([i for i in range(1000)], first_text_field[:1000], second_text_field[:1000], explanations[:1000]):

        prompt_template = get_demonstrations(demo_first_text_field, demo_second_text_field, demo_answers, demo_explanations, dataset_name=dataset_name)
        prompt_template += esnli_template(idx, first, second, dataset_name, model_name)
        generator_explanation = prompt_template_model_generation(model, tokenizer, prompt_template)
        save_explanation(f"eSNLI_generator_explanation_{model_name}", idx, generator_explanation)


def healthFC_generator_explanation(dataset_name, model_name, model, tokenizer):
    first_text_field, second_text_field, demo_answers, demo_explanations = load_dataset("healthFC_train")
    questions, documents, answers, explanations = load_dataset("healthFC_val")

    for idx, question, document, explanation in zip([i for i in range(len(questions))], questions, documents, explanations):
        prompt_template = get_demonstrations(first_text_field, second_text_field, demo_answers, demo_explanations, num_shot=3, dataset_name=dataset_name)
        prompt_template += healthFC_template(idx, question, document, dataset_name, model_name)

        generator_explanation = prompt_template_model_generation(model, tokenizer, prompt_template)
        save_explanation(f"healthFC_generator_explanation_{model_name}", idx, generator_explanation)


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

    if dataset == "ECQA":
        ECQA_generator_explanation(dataset, model_name.split("/")[1], model, tokenizer)
    elif dataset == "eSNLI":
        eSNLI_generator_explanation(dataset, model_name.split("/")[1], model, tokenizer)
    elif dataset == "healthFC":
        healthFC_generator_explanation(dataset, model_name.split("/")[1], model, tokenizer)
