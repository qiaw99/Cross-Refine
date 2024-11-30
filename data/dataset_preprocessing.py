import argparse
import os
import ast

import pandas as pd


def preprocess_ECQA():
    df = pd.read_csv(f"{os.getcwd()}/ECQA/cqa_data_train.csv")

    texts = df["q_text"]
    positive_explanations = df["taskA_pos"]
    negative_explanations = df["taskA_neg"]
    free_flow_explanations = df["taskB"]

    choices = []
    answers = []
    idx = [i for i in range(1, 6)]

    for i in range(len(texts)):
        choice = []
        for j in idx:
            choice.append(df["q_op" + str(j)][i])
        ans = df["q_ans"][i]
        answers.append(choice.index(ans))
        choices.append("-".join(choice))

    name_dict = {
        "texts": texts,
        "choices": choices,
        "answers": answers,
        "positive_explanations": positive_explanations,
        "negative_explanations": negative_explanations,
        "free_flow_explanations": free_flow_explanations
    }

    df = pd.DataFrame(name_dict)
    df.to_csv(f'{os.getcwd()}/ECQA/ECQA_train.csv', encoding='utf-8')


def preprocess_eSNLI():
    df_train = pd.read_csv(f"{os.getcwd()}/eSNLI/original_esnli/esnli_dev.csv")
    df_val = pd.read_csv(f"{os.getcwd()}/eSNLI/original_esnli/esnli_test.csv")

    train_labels = df_train["gold_label"]
    train_sentence1 = df_train["Sentence1"]
    train_sentence2 = df_train["Sentence2"]
    train_explanations = df_train["Explanation_1"]

    val_labels = df_val["gold_label"]
    val_sentence1 = df_val["Sentence1"]
    val_sentence2 = df_val["Sentence2"]
    val_explanations = df_val["Explanation_1"]

    train_name_dict = {
        "label": train_labels,
        "sentence1": train_sentence1,
        "sentence2": train_sentence2,
        "explanation": train_explanations
    }

    val_name_dict = {
        "label": val_labels,
        "sentence1": val_sentence1,
        "sentence2": val_sentence2,
        "explanation": val_explanations
    }

    df = pd.DataFrame(train_name_dict)
    df.to_csv(f'{os.getcwd()}/eSNLI/eSNLI_train.csv', encoding='utf-8')

    df = pd.DataFrame(val_name_dict)
    df.to_csv(f'{os.getcwd()}/eSNLI/eSNLI_val.csv', encoding='utf-8')


def preprocess_healthFC():
    # Read the original data
    df = pd.read_csv(f"{os.getcwd()}/healthFC/original_healthFC/healthFC_annotated.csv")

    # Extract only the English data and do the conversion using the following fields:
    # questions, documents, answers, explanations
    # where answers have the following label mapping wrt the original annotations:
    id2label = {0: "yes", 1: "unknown", 2: "no"}
    training_data = {"questions": [], "documents": [], "answers": [], "explanations": []}
    validation_data = {"questions": [], "documents": [], "answers": [], "explanations": []}
    demo_data = {"questions": [], "documents": [], "answers": [], "explanations": []}

    demo_threshold = 50 # We set aside 50 samples to be used as demostrations to avoid showing the sample that needs to be explained among the demonstrations.
    train_threshold = demo_threshold + round((len(df["en_claim"])-demo_threshold)/2)

    for idx in range(len(df["en_claim"])):
        question = df["en_claim"][idx]
        # document = df["en_text"][idx] # Note that original documents are too long! Hence, we store only the evidence sentences as a new document.
        sentences = ast.literal_eval(df["en_sentences"][idx])
        evidence = ast.literal_eval(df["en_ids"][idx])
        document = ""
        for sentence_id in evidence:
            document += sentences[sentence_id] + " "
        document = document.strip()
        answer = id2label[df["label"][idx]]
        explanation = df["en_explanation"][idx]
        if idx < demo_threshold:
            data = demo_data
        elif idx < train_threshold:
            data = training_data
        else:
            data = validation_data
        data["questions"].append(question)
        data["documents"].append(document)
        data["answers"].append(answer)
        data["explanations"].append(explanation)

    training_df = pd.DataFrame.from_dict(training_data)
    validation_df = pd.DataFrame.from_dict(validation_data)
    demo_df = pd.DataFrame.from_dict(demo_data)

    demo_df.to_csv(f"{os.getcwd()}/healthFC/healthFC_demo.csv", sep=",", encoding="utf-8")
    training_df.to_csv(f"{os.getcwd()}/healthFC/healthFC_train.csv", sep=",", encoding="utf-8")
    validation_df.to_csv(f"{os.getcwd()}/healthFC/healthFC_val.csv", sep=",", encoding="utf-8")


def main(dataset):

    if dataset == "ECQA":
        preprocess_ECQA()
    elif dataset == "eSNLI":
        preprocess_eSNLI()
    elif dataset == "healthFC":
        preprocess_healthFC()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="ECQA",
        choices=["ECQA", "eSNLI", "healthFC"],
        help="Identify which dataset to preprocess",
    )
    args = vars(parser.parse_args())
    main(**args)
