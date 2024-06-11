import argparse
import os
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    SummarizationPipeline,
    AutoConfig,
)
from transformers import logging
import warnings
import pandas as pd


# Suppress warnings and set logging to error
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Define the model names
model_names = {
    "code_trans_t5_sum": "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune",
    "code_trans_t5_doc": "SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune",
    "code_t5_base_sum": "Salesforce/codet5-base-multi-sum",
    "code_t5p_base_sum": "Paul-B98/codet5p_220m_py_sum",
    "pile_t5_large_sum": "lintang/pile-t5-large-codexglue",
}

DATASET = ["mce", "mcsn"]

LEVEL = "method"


def load_data(file_path, dataset):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if dataset != "mcsn":
        df = pd.read_json(file_path, lines=True).set_index("class_id")
    else:
        df = pd.read_json(file_path, lines=True).set_index("repo_name")
    return df


def load_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_name
            if model_name != "Paul-B98/codet5p_220m_py_sum"
            else "Salesforce/codet5p-220m"
        ),
        skip_special_tokens=True,
        legacy=False,
        padding=True,
    )
    config = AutoConfig.from_pretrained(model_name)
    return model, tokenizer, config


def create_predictions(df, pipeline):
    tqdm.pandas()
    df["pred_summary"] = df["method_code"].progress_apply(
        lambda x: pipeline(x, do_sample=True, num_beams=5)[0]["summary_text"]
    )
    return df


def save_predictions(df, model_dir, level, dataset):
    df = df.reset_index()
    output_file_path = model_dir / f"{level}-level-{dataset}-pred.jsonl"
    df.to_json(output_file_path, orient="records", lines=True)
    print(f"Predictions saved to {output_file_path}")


def inference(model_key, data_dir, dataset):
    # Check if the data directory exists
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Directory {data_dir} does not exist.")

    PREPROCESSED_DATA_DIR = data_dir / "preprocessed"
    PREDICTED_DATA_DIR = data_dir / "predicted"

    file_path = PREPROCESSED_DATA_DIR / f"{LEVEL}-level-{dataset}.jsonl"
    df = load_data(file_path, dataset)

    model_name = model_names.get(model_key)
    if not model_name:
        raise ValueError(
            f"Invalid model key. Choose from: {', '.join(model_names.keys())}"
        )

    model, tokenizer, config = load_model_and_tokenizer(model_name)

    pipeline = SummarizationPipeline(model, tokenizer, config, device="cuda:3")

    df = create_predictions(df, pipeline)

    MODEL_DIR = PREDICTED_DATA_DIR / model_name.split("/")[-1]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    save_predictions(df, MODEL_DIR, LEVEL, dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions using a PLM for Code Summarization task."
    )
    parser.add_argument("data_dir", type=Path, help="Directory with data")
    parser.add_argument("dataset", choices=DATASET, help="Dataset to use")
    parser.add_argument(
        "model_key", choices=model_names.keys(), help="Key of the model from the list"
    )
    args = parser.parse_args()

    inference(args.model_key, args.data_dir, args.dataset)


if __name__ == "__main__":
    main()
