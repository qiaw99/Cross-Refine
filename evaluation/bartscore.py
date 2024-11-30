"""
bartscore evaluation
"""

import numpy as np
from bart_score import BARTScorer
from experiments.utils import load_rationale_and_json_files, load_json


def bartscore_calculation(predictions, references):
    scores = bart_scorer.score(predictions, references,
                               batch_size=4)  # generation scores from the first list of texts to the second list of texts.
    print(sum(scores) / len(scores))


if __name__ == "__main__":
    # folder_path = "explanation_generator"
    folder_path = "self_refinement"
    # folder_path = "ablation_study"
    # folder_path = "refined_explanation"

    # field = "explanation"
    field = "self_refined_explanation"
    # field = "refined_explanation"

    json_files, ecqa_explanations, esnli_explanations, healthfc_explanations = load_rationale_and_json_files(
        f"../results/{folder_path}/")
    print(json_files)

    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')

    for json_file in json_files:
        if json_file.startswith("healthFC"):
            json_ls = load_json(f"../results/{folder_path}/{json_file}")
            idx = []
            generated_explanation = []
            for i, ex in enumerate(json_ls):
                if ex[field] != "":
                    generated_explanation.append(ex[field])
                    idx.append(ex["idx"])
            print(json_file, end="")
            bartscore_calculation(generated_explanation, list(np.array(healthfc_explanations)[np.array(idx)]))
        elif json_file.startswith("ECQA"):
            json_ls = load_json(f"../results/{folder_path}/{json_file}")
            idx = []
            generated_explanation = []
            for i, ex in enumerate(json_ls):
                if ex[field] != "":
                    generated_explanation.append(ex[field])
                    idx.append(ex["idx"])
            print(json_file, end="")
            # print(idx, end="")
            print(idx)
            bartscore_calculation(generated_explanation, list(np.array(ecqa_explanations)[np.array(idx)]))
        elif json_file.startswith("eSNLI"):
            json_ls = load_json(f"../results/{folder_path}/{json_file}")
            idx = []
            generated_explanation = []
            for i, ex in enumerate(json_ls):
                if ex[field] != "" and not ex[field].startswith("1"):
                    generated_explanation.append(ex[field])
                    idx.append(ex["idx"])
            print(json_file, end="")
            bartscore_calculation(generated_explanation, list(np.array(esnli_explanations)[np.array(idx)]))

