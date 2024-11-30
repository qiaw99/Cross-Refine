from bleurt import score
from experiments.utils import load_rationale_and_json_files, load_json
import numpy as np


def bleurtscore_calculation(predictions, references):
    scores = scorer.score(references=references, candidates=predictions)

    print(sum(scores)/len(scores))


if __name__ == "__main__":
    checkpoint = "bleurt/test_checkpoint"
    scorer = score.BleurtScorer()
    # folder_path = "explanation_generator"
    # folder_path = "self_refinement"
    # folder_path = "ablation_study"
    folder_path = "refined_explanation"

    # field = "explanation"
    # field = "self_refined_explanation"
    field = "refined_explanation"

    json_files, ecqa_explanations, esnli_explanations, healthfc_explanations = load_rationale_and_json_files(
        f"../results/{folder_path}")

    for json_file in json_files:
        if json_file.startswith("healthFC"):
            json_ls = load_json(f"../results/{folder_path}/healthFC/{json_file}")
            idx = []
            generated_explanation = []
            for i, ex in enumerate(json_ls):
                if ex[field] != "":
                    generated_explanation.append(ex[field])
                    idx.append(ex["idx"])
            print(json_file, end="")
            bleurtscore_calculation(generated_explanation, list(np.array(healthfc_explanations)[np.array(idx)]))
        elif json_file.startswith("ECQA"):
            json_ls = load_json(f"../results/{folder_path}/{json_file}")
            idx = []
            generated_explanation = []
            for i, ex in enumerate(json_ls):
                if ex[field] != "":
                    generated_explanation.append(ex[field])
                    idx.append(ex["idx"])
            print(json_file, end="")
            bleurtscore_calculation(generated_explanation, list(np.array(ecqa_explanations)[np.array(idx)]))
        elif json_file.startswith("eSNLI"):
            json_ls = load_json(f"../results/{folder_path}/{json_file}")
            idx = []
            generated_explanation = []
            for i, ex in enumerate(json_ls):
                if ex[field] != "" and not ex[field].startswith("1"):
                    generated_explanation.append(ex[field])
                    idx.append(ex["idx"])
            print(json_file, end="")
            bleurtscore_calculation(generated_explanation, list(np.array(esnli_explanations)[np.array(idx)]))
