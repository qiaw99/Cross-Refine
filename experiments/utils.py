import json
from os import listdir
from os.path import join, isfile

import pandas as pd


def get_prediction_and_label_by_idx(idx, model_name, dataset_name):
    f = open(f"../results/prediction/{dataset_name}_prediction_{model_name.split('/')[1]}_20shots.json")
    data = json.load(f)
    return data[idx]["prediction"], data[idx]["label"]


def load_json(path):
    jsonFile = open(path, "r")

    jsonContent = jsonFile.read()
    content_ls = json.loads(jsonContent)
    jsonFile.close()

    return content_ls


def clean_up(text):
    if "\n" in text and not text.startswith("\n"):
        text = text[:text.index("\n")]
    if "." in text:
        text = text[:text.rindex(".")]

    return text.strip()
  

def load_gold_rationale(dataset_name):
    if dataset_name == "ECQA":
        df = pd.read_csv("../data/ECQA/ECQA_val.csv")
        return list(df["free_flow_explanations"])
    elif dataset_name == "eSNLI":
        df = pd.read_csv("../data/eSNLI/eSNLI_val.csv")
        return list(df["explanation"])
    elif dataset_name == "healthFC":
        df = pd.read_csv("../data/healthFC/de/healthFC_val.csv")
        return list(df["explanations"])
    else:
        raise ValueError(f"Unknow dataset: {dataset_name}")


def load_rationale_and_json_files(explanation_path_prefix="../results/explanation_generator"):
    json_files = [f for f in listdir(explanation_path_prefix) if
                  isfile(join(explanation_path_prefix, f)) and f.endswith(".json")]
    json_files += [f for f in listdir(f"{explanation_path_prefix}/") if
                  isfile(join(f"{explanation_path_prefix}/", f)) and f.endswith(".json")]

    ecqa_explanations = load_gold_rationale("ECQA")
    esnli_explanations = load_gold_rationale("eSNLI")
    healthfc_explanations = load_gold_rationale("healthFC")

    return json_files, ecqa_explanations, esnli_explanations, healthfc_explanations
