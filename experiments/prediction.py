import json
import random
import re

import argparse

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from guided_decoding.gd_logit_processor import GuidedParser, GuidedDecodingLogitsProcessor
from guided_decoding.grammar import ECQA_GRAMMAR, eSNLI_grammar, healthFC_grammar


def load_dataset(dataset_name="ECQA_val"):
    """
    Dataset loading
    :param dataset_name: dataset name
    :return:
    """
    if "ECQA" in dataset_name:
        df = pd.read_csv(f"data/ECQA/{dataset_name}.csv")
        return list(df["texts"]), list(df["choices"]), list(df["answers"]), list(df["free_flow_explanations"])
    elif "eSNLI" in dataset_name:
        df = pd.read_csv(f"data/eSNLI/{dataset_name}.csv")
        return list(df["label"]), list(df["sentence1"]), list(df["sentence2"]), list(df["explanation"])
    elif "healthFC" in dataset_name:
        df = pd.read_csv(f"data/healthFC/{dataset_name}.csv")
        return list(df["questions"]), list(df["documents"]), list(df["answers"]), list(df["explanations"])


def get_demonstrations_for_prediction(demo_texts, demo_answers, demo_choices=None, num_shot=5, dataset_name="ECQA"):
    random_idx_list = [random.randint(0, len(demo_texts) - 1) for i in range(num_shot)]
    demo_prompt = ""

    if dataset_name == "ECQA":
        for _id in random_idx_list:
            demo_prompt += f"Question: {demo_texts[_id]}\n"

            for c, ch in zip([i for i in range(1, 6)], demo_choices[_id].split("-")):
                demo_prompt += f"Choice {c}: {ch}\n"
            # demo_prompt += f"Prediction: {demo_choices[_id].split('-')[int(demo_answers[_id])]}\n\n"
            demo_prompt += f"Prediction: {int(demo_answers[_id])+1}\n\n"
    elif dataset_name == "eSNLI":
        for _id in random_idx_list:
            demo_prompt += f"Premise: {demo_texts[_id]}\n"
            demo_prompt += f"Hypothesis: {demo_choices[_id]}\n"
            demo_prompt += f"Label: {demo_answers[_id]}\n\n"
    elif dataset_name == "healthFC":
        for _id in random_idx_list:
            demo_prompt += f"Question: {demo_texts[_id]}\n"
            demo_prompt += f"Document: {demo_choices[_id]}\n"
            demo_prompt += f"Label: {demo_answers[_id]}\n\n"
    return demo_prompt


def get_ECQA_predictions():
    demo_texts, demo_choices, demo_answers, demo_explanations = load_dataset("ECQA_train")
    texts, choices, answers, explanations = load_dataset()

    predictions = []

    for idx, text, choice, answer, explanation in zip([i for i in range(len(texts))], texts, choices, answers,
                                                      explanations):
        prompt_template = (
            f"Given a commonsense question and 5 possible answer choices, please select the answer choice that best "
            f"answers the question based on common sense and general world knowledge.\n")

        prompt_template += get_demonstrations_for_prediction(demo_texts, demo_choices, demo_answers)

        prompt_template += f"Question: {text}\n"

        for m, n in zip([i for i in range(1, 6)], choice.split("-")):
            prompt_template += f"Choice {m}: {n}\n"
        prompt_template += "Prediction:"

        # print(prompt_template)

        input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(device)

        parser = GuidedParser(ECQA_GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
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
            match = re.findall(r'\d+', result)
            result = list(map(int, match))[-1]
            predictions.append(result)

            print(f"Idx: {idx}; prediction: {choice.split('-')[int(result)-1]}; label: {choice.split('-')[int(answer)]}")
        except:
            splits_string = tokenizer.decode(output[0]).split(prompt_template)
            match = re.findall(r'\d+', splits_string[0])
            if match:
                predictions.append(list(map(int, match))[-1])

    output = []
    for i in range(len(predictions)):
        output.append({
            "idx": i,
            "prediction": choices[i].split('-')[int(predictions[i])-1],
            "label": choices[i].split('-')[int(answers[i])]
        })
    jsonFile = open(f"results/prediction/ECQA_prediction_{model_name.split('/')[1]}.json", "w")
    jsonString = json.dumps(output)
    jsonFile.write(jsonString)
    jsonFile.close()


def get_eSNLI_predictions():
    labels, sentence1, sentence2, explanations = load_dataset("eSNLI_val")
    demo_labels, demo_sentence1, demo_sentence2, demo_explanation = load_dataset("eSNLI_train")

    predictions = []

    for idx, label, s1, s2, explanation in zip([i for i in range(len(labels))], labels, sentence1, sentence2, explanations):
        prompt_template = (
            f"You are provided with a premise and a hypothesis. Your task is to determine the relationship between "
            f"the premise and the hypothesis. The possible relationships are: Neutral: The hypothesis is neither "
            f"supported nor contradicted by the premise. Entailment: The hypothesis logically follows from the "
            f"premise. Contradiction: The hypothesis is logically opposite to what the premise states. Based on "
            f"the given premise and hypothesis, choose the most appropriate label: neutral, entailment, "
            f"or contradiction.\n\n")

        prompt_template += get_demonstrations_for_prediction(demo_sentence1, demo_labels, demo_sentence2, dataset_name="eSNLI")

        prompt_template += f"Premise: {s1}\n"
        prompt_template += f"Hypothesis: {s2}\n"
        prompt_template += f"Label:"

        # print(prompt_template)

        input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(device)

        parser = GuidedParser(eSNLI_grammar, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
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

            # post-processing for mixtral
            ls = result.split(" ")
            for (_idx, i) in enumerate(ls):
                if "<s>" in i:
                    ls[_idx] = i.split("<s>")[0]
            ls = [i for i in ls if i != '']
            result = " ".join(ls)

            predictions.append(result)

            print(f"Idx: {idx}; prediction: {result}; label: {label}")
        except:
            predictions.append(tokenizer.decode(output[0]).split(prompt_template))

    output = []
    for i in range(len(predictions)):
        output.append({
            "idx": i,
            "prediction": predictions[i],
            "label": labels[i]
        })
    jsonFile = open(f"results/prediction/eSNLI_prediction_{model_name.split('/')[1]}.json", "w")
    jsonString = json.dumps(output)
    jsonFile.write(jsonString)
    jsonFile.close()


def get_healthFC_predictions():
    demo_questions, demo_documents, demo_answers, demo_explanations = load_dataset("healthFC_demo")
    questions, documents, answers, explanations = load_dataset("healthFC_val")

    predictions = []

    for idx, question, document, answer, explanation in zip([i for i in range(len(questions))], questions, documents, answers,
                                                      explanations):
        prompt_template = (
            f"Given the question and the corresponding document text, please select an answer from one of the following three options: yes, no or unknown.\n")

        prompt_template += get_demonstrations_for_prediction(demo_questions, demo_answers, demo_documents, dataset_name="healthFC", num_shot=1)

        prompt_template += f"Document: {document}\n:"
        prompt_template += f"Question: {question}\n"
        prompt_template += "Label:"

        input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(device)

        parser = GuidedParser(healthFC_grammar, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
        guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

        with torch.no_grad():
            output = model.greedy_search(
                input_ids=input_ids,
                logits_processor=guided_preprocessor,
                eos_token_id=parser.eos_token,
                pad_token_id=model.config.pad_token_id,
            )
        decoded_output = tokenizer.decode(output[0])
        try:
            result = decoded_output.split(prompt_template)[1].split(" [e]")[0].strip()
            predictions.append(result)
            print(f"Idx: {idx}; prediction: {result}; label: {answer}")
        except:
            result = decoded_output[len(prompt_template):].strip()
            if result not in ["yes", "no", "unknown"]:
                result = "unknown"
            predictions.append(result)
            print(f"Could not decode the generated output, default to removing the prompt text and matching to one of the labels: {result}")

    output = []
    for i in range(len(predictions)):
        output.append({
            "idx": i,
            "prediction": predictions[i],
            "label": answers[i]
        })
    jsonFile = open(f"results/prediction/healthFC_prediction_{model_name.split('/')[1]}.json", "w")
    jsonString = json.dumps(output)
    jsonFile.write(jsonString)
    jsonFile.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model name, e.g. TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ")
    parser.add_argument("--dataset", help="dataset name, e.g. eSNLI, ECQA or healthFC")
    args = parser.parse_args()
    model_name = args.model_name
    dataset = args.dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if dataset == "eSNLI":
        get_eSNLI_predictions()
    elif dataset == "ECQA":
        get_ECQA_predictions()
    elif dataset == "healthFC":
        get_healthFC_predictions()
    else:
        raise ValueError("Unknown dataset. Can be one of the following: eSNLI, ECQA, healthFC.")
