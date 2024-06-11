import argparse
import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from evaluate import load
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import torch.nn.functional as F
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.proxies = {"https": "http://34e9515e90e14e90:9d35c556ec546bc6@135.181.81.30:3128"}
    # session.verify = False
    return session

class Metrics(Enum):
    ROUGE = "ROUGE-L"
    BLEU = "BLEU-4"
    METEOR = "METEOR"
    BERTScore = "BERTScore"
    BLEURT = "BLEURT"
    SIDE_TRUE = "SIDE_true"
    SIDE_PRED = "SIDE_pred"


DATASETS = ["mce", "mcsn"]
LEVELS = ["method", "class", "repo"]

def print_results(model_dir):
    """
    Prints the results of the evaluation.

    Args:
        model_dir (Path): The path to the file containing the results.

    Returns:
        None
    """
    dfs = []
    for file_path in model_dir.glob('*eval.json'):
        with file_path.open('r') as file:
            data = json.load(file)
            if 'mcsn' in str(file_path):
                for key in data.keys():
                    sub_data = data[key]
                    sub_data['Name'] = '/'.join(str(file_path).split('/')[-2:]) + '/' + key
                    dfs.append(pd.DataFrame([sub_data]))
            else:
                data['Name'] = '/'.join(str(file_path).split('/')[-2:])
                dfs.append(pd.DataFrame([data]))

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.set_index("Name", inplace=True)
    markdown_table = combined_df.to_markdown()
    print(markdown_table)


def compute_side(tokenizer, model, code, summary):
    """
    Compute the similarity between a given code and a summary using a pre-trained model.

    Args:
        tokenizer (Tokenizer): The tokenizer used to tokenize the code and summary.
        model (Model): The pre-trained model used to compute the sentence embeddings.
        code (str): The code to be compared.
        summary (str): The summary to be compared.

    Returns:
        float: The cosine similarity between the code and summary.
    """

    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    pair = [code, summary]
    encoded_input = tokenizer(
        pair, padding=True, truncation=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sim = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()
    return sim


def save_metrics_eval(df, model_dir, level, dataset, num_few_shot=None, k=None):
    """
    Save the evaluation metrics for the model.

    Args:
        df (DataFrame): The dataframe containing the evaluation metrics.
        model_dir (Path): The directory where the model is saved.
        level (str): The level of evaluation.
        dataset (str): The dataset for evaluation.
        num_few_shot (int, optional): The number of few-shot examples used in evaluation. Defaults to None.
    """
    if dataset != "mcsn":
        total_avg_metrics = {
            metric.value: df[metric.value].mean() for metric in Metrics
        }
    else:
        total_avg_metrics = {
            "total": {metric.value: df[metric.value].mean() for metric in Metrics}
        }
        repo_avg_metrics = {
            repo_name: {
                metric.value: df.loc[df["repo_name"] == repo_name, metric.value].mean()
                for metric in Metrics
            }
            for repo_name in df["repo_name"].unique()
        }
        total_avg_metrics.update(repo_avg_metrics)

    if num_few_shot is not None:
        if k is not None:
            file_name = (
                model_dir
                / f"{level}-level-{dataset}-few-shot-{num_few_shot}-context-{k}-eval.json"
            )
        else:
            file_name = (
                model_dir
                / f"{level}-level-{dataset}-few-shot-{num_few_shot}-eval.json"
            )
    else:
        file_name = model_dir / f"{level}-level-{dataset}-eval.json"
    with open(file_name, "w") as file:
        json.dump(total_avg_metrics, file, indent=4)


def evaluate_model(model_dir, level, dataset, few_shot):
    """
    Evaluates a model's predictions on a given level of summarization.

    Args:
        model_dir (Path): The directory containing the model predictions file.
        level (str): The level of summarization ('method' by default).
        dataset (str): The dataset to evaluate on.
        few_shot (bool): Whether to use few-shot examples or not.

    Raises:
        FileNotFoundError: If the model predictions file does not exist.

    Returns:
        None
    """
    # Check for GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This script requires a GPU to run.")

    if few_shot:
        # Find all paths at model_dir with NUM
        file_paths = [
            file_path
            for file_path in model_dir.glob(
                f"*{level}-level-{dataset}-few-shot-*-pred.jsonl"
            )
            if file_path.exists()
        ]
        if not file_paths:
            raise FileNotFoundError(
                f"No file found at {model_dir} with pattern {level}-level-{dataset}-few-shot-*.jsonl"
            )
        file_paths.sort(key=lambda x: int(x.stem.split("-")[-2]))
    else:
        file_paths = [model_dir / f"{level}-level-{dataset}-pred.jsonl"]
        if level == "class":
            file_paths.append(model_dir / f"skeleton-level-{dataset}-pred.jsonl")
    # remove already computed
    # new_file_paths = []
    # for file_path in file_paths:
    #     metrics_file_path = file_path.parent / f"{file_path.stem}-metrics.jsonl"
    #     if not metrics_file_path.exists():
    #         new_file_paths.append(file_path)
    #     else:
    #         print(f"File {metrics_file_path} already exists. Skipping...")
    # file_paths = new_file_paths

    # load metrics
    def _compute_side_partial(args):
        return compute_side(side_tokenizer, side_model, *args)

    side_model_dir = Path("../models/side")
    side_tokenizer = AutoTokenizer.from_pretrained(side_model_dir)
    side_model = AutoModel.from_pretrained(side_model_dir).to("cuda:0")

    bleurt = load("bleurt", "BLEURT-20", module_type="metric")
    bertscore = load("bertscore")
    bleu = load("bleu")
    rouge = load("rouge")
    meteor = load("meteor")
    print(f"Evaluating {model_dir}...")
    for file_path in file_paths:
        num_few_shot = int(file_path.stem.split("-")[-4 if level == 'repo' else -2]) if few_shot else None
        print(f"Evaluating {file_path.stem}...")
        df = pd.read_json(file_path, lines=True)

        true_column_name = "method_summary"
        pred_column_name = "pred_summary"

        true_summaries = df[true_column_name].to_list()
        pred_summaries = df[pred_column_name].to_list()
        assert len(true_summaries) == len(pred_summaries)

        for metric in Metrics:
            df[metric.value] = None

        # compute metrics
        print("Computing BERTScore...")
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
        bertscore_res = bertscore.compute(
            predictions=pred_summaries,
            references=true_summaries,
            model_type="microsoft/deberta-xlarge-mnli",
            device=device,
        )
        df[Metrics.BERTScore.value] = bertscore_res["f1"]

        print("Computing BLEURT...")
        bleurt_res = bleurt.compute(
            predictions=pred_summaries, references=true_summaries
        )
        df[Metrics.BLEURT.value] = bleurt_res["scores"]

        print("Computing SIDE...")
        tqdm.pandas()
        df[Metrics.SIDE_TRUE.value] = df[
            ["method_code", true_column_name]
        ].progress_apply(_compute_side_partial, axis=1)
        df[Metrics.SIDE_PRED.value] = df[
            ["method_code", pred_column_name]
        ].progress_apply(_compute_side_partial, axis=1)

        print("Computing BLEU-4, ROUGE-L, and METEOR...")
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            true_summary = row[true_column_name]
            pred_summary = row[pred_column_name]

            try:
                bleu_res = bleu.compute(
                    predictions=[pred_summary], references=[[true_summary]]
                )
            except ZeroDivisionError:
                bleu_res = {"bleu": 0.0}
            rouge_res = rouge.compute(
                predictions=[pred_summary], references=[true_summary]
            )
            meteor_res = meteor.compute(
                predictions=[pred_summary], references=[true_summary]
            )

            df.at[index, Metrics.ROUGE.value] = rouge_res["rougeL"]
            df.at[index, Metrics.BLEU.value] = bleu_res["bleu"]
            df.at[index, Metrics.METEOR.value] = meteor_res["meteor"]

        # save metrics
        df = df.reset_index()
        temp_level = file_path.stem.split("-")[0]
        k = int(file_path.stem.split("-")[-2]) if temp_level == 'repo' else None
        if few_shot:
            if k is not None:
                pred_metrics_path = (
                    model_dir
                    / f"{temp_level}-level-{dataset}-few-shot-{num_few_shot}-context-{k}-pred-metrics.jsonl"
                )
            else:
                pred_metrics_path = (
                    model_dir
                    / f"{temp_level}-level-{dataset}-few-shot-{num_few_shot}-pred-metrics.jsonl"
                )
        else:
            pred_metrics_path = (
                model_dir / f"{temp_level}-level-{dataset}-pred-metrics.jsonl"
            )
        df.to_json(
            pred_metrics_path,
            orient="records",
            lines=True,
        )
        save_metrics_eval(df, model_dir, temp_level, dataset, num_few_shot, k)



def main():
    """
    Evaluate model predictions.

    Args:
        model_dir (Path): Directory with model predictions file (.jsonl).
        --level (str, optional): Level of summarization (default: "method").

    Raises:
        NotADirectoryError: If the specified directory does not exist.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument(
        "model_dir", type=Path, help="Directory with model predictions file (.jsonl)"
    )
    parser.add_argument("dataset", choices=DATASETS, help="Dataset to use")
    parser.add_argument(
        "--level", default="method", choices=LEVELS, help="Level of summarization (default: method)"
    )
    parser.add_argument(
        "--few-shot", default=False, help="Use few-shot examples (default: False)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise NotADirectoryError(f"Directory {args.model_dir} does not exist.")
    if args.level == "class" and (args.few_shot or args.dataset == "mcsn"):
        raise ValueError("Modified CodeSearchNet or Few-shot examples are not available for class level.")
    if args.level == "repo" and args.dataset == "mce":
        raise ValueError("Modified ClassEval is not available for repo level.")

    configure_http_backend(backend_factory=backend_factory)
    evaluate_model(args.model_dir, args.level, args.dataset, args.few_shot)
    print_results(args.model_dir)


if __name__ == "__main__":
    main()
