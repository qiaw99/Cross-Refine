"""
tiger score evaluation
"""

import json
import os

from experiments.critic_feedback_explanation_generation import get_context
from experiments.utils import load_rationale_and_json_files, load_json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tigerscore import TIGERScorer


def dataset_input_context(first, second, dataset_name):
    input_context = ""
    instruction = ""
    if dataset_name == "ECQA":
        instruction = "Evaluate the generated explanation based on the given question and choices."
        input_context += f"question: {first}\nchoices: {second}"
    elif dataset_name == "eSNLI":
        instruction = "Evaluate the generated explanation based on the given premise and hypothesis."
        input_context += f"premise: {first}\nhypothesis: {second}"
    else:
        pass

    return input_context, instruction


if __name__ == "__main__":
    # path = "self_refinement"
    # path = "ablation_study"
    path = "refined_explanation"

    # field = "self_refined_explanation"
    field = "refined_explanation"

    json_files, ecqa_explanations, esnli_explanations, healthfc_explanations = load_rationale_and_json_files(
        f"../results/{path}")
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", quantized=True)  # 4 bit quantization on GPU
    res = []

    for json_file in json_files:
        scores = []

        if json_file.startswith("healthFC"):
            json_ls = load_json(f"../results/{path}/healthFC/{json_file}")
        else:
            json_ls = load_json(f"../results/{path}/{json_file}")

        generated_explanation = []
        idx = []

        print(json_file)

        for ex in json_ls:
            if ex[field] != "":
                generated_explanation.append(ex[field])
                idx.append(ex["idx"])

        dataset_name = json_file.split("_")[0]

        for i, generation in enumerate(generated_explanation[:100]):
            first, second = get_context(idx[i], dataset_name)
            input_context, instruction = dataset_input_context(first, second, dataset_name)
            results = scorer.score([instruction], [generation], [input_context])
            if results[0]["score"] is not None:
                scores.append(results[0]["score"])
            else:
                print(results[0])

        print(f"{json_file}: {scores}")

        try:
            res.append({
                "name": json_file,
                "score": sum(scores) / len(scores)
            })
        except:
            res.append({
                "name": json_file,
                "score": scores
            })

        jsonFile = open(
            f"./tiger_score_{path}.json", "w")
        jsonString = json.dumps(res)
        jsonFile.write(jsonString)
        jsonFile.close()
