import json
from os import listdir
from os.path import join, isfile

from experiments.utils import load_json

from sentence_transformers import SentenceTransformer, util

combinations = ["ECQA", "eSNLI", "healthFC"]

results = []

for combi in combinations:
    suggestion_prefix = f"../results/feedback_suggestion/{combi}/"
    if combi == "healthFC":
        refinement_prefix = f"../results/refined_explanation/{combi}/"
    else:
        refinement_prefix = f"../results/refined_explanation/"
    json_files = [f for f in listdir(suggestion_prefix) if
                  isfile(join(suggestion_prefix, f)) and f.endswith(".json") and f.startswith(combi)]
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to("cuda")

    for json_f in json_files:
        feedback_and_suggestions = load_json(f"{suggestion_prefix}{json_f}")

        refined_json_file = f"{'_'.join(json_f.split('_')[:-3])}_refined_explanation.json"
        # refined_explanation = load_json(f"{refinement_prefix}{refined_json_file}")
        try:

            refined_explanation = load_json(f"{refinement_prefix}{refined_json_file}")
            refined_explanation_ls = [j["refined_explanation"] for j in refined_explanation if j["refined_explanation"] != ""]

            idx = [j["idx"] for j in refined_explanation if j["refined_explanation"] != ""]

            refined_explanation_embeddings = model.encode(refined_explanation_ls, convert_to_tensor=True).to("cuda")

            if combi == "healthFC":
                suggestions = [item["suggestion"] for item in feedback_and_suggestions if item["idx"] in idx]
            else:
                suggestions = [item["suggested_explanation"] for item in feedback_and_suggestions if item["idx"] in idx]

            explanations = [item["initial_explanation"] for item in feedback_and_suggestions if item["idx"] in idx]
            feedbacks = [item["feedback"] for item in feedback_and_suggestions if item["idx"] in idx]

            suggestion_embeddings = model.encode(suggestions, convert_to_tensor=True).to("cuda")
            explanation_embeddings = model.encode(explanations, convert_to_tensor=True).to("cuda")
            feedback_embeddings = model.encode(feedbacks, convert_to_tensor=True).to("cuda")

            suggestion_cosine_scores = []
            explanation_cosine_scores = []
            feedback_cosine_scores = []

            for i in range(len(refined_explanation_embeddings)):
                suggestion_cosine_scores.append(util.cos_sim(refined_explanation_embeddings[i], suggestion_embeddings[i]).cpu().detach().item())
                explanation_cosine_scores.append(util.cos_sim(refined_explanation_embeddings[i], explanation_embeddings[i]).cpu().detach().item())
                feedback_cosine_scores.append(util.cos_sim(refined_explanation_embeddings[i], feedback_embeddings[i]).cpu().detach().item())
            results.append({
                "file_name": refined_json_file,
                "explanation_cos_sim": sum(explanation_cosine_scores) / len(explanation_cosine_scores),
                "feedback_cos_sim": sum(feedback_cosine_scores) / len(feedback_cosine_scores),
                "suggestion_cos_sim": sum(suggestion_cosine_scores)/len(suggestion_cosine_scores)
            })
        except:
            print(refined_json_file)

    jsonFile = open("results/cos_similarity_scores.json", "w")
    jsonString = json.dumps(results)
    jsonFile.write(jsonString)
    jsonFile.close()

