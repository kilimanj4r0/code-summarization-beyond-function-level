import argparse
import os
import torch
import warnings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# Suppress warnings and set logging to error
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Define the model names
model_names = {
    "dsc-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "dsc-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "dsc-33b": "deepseek-ai/deepseek-coder-33b-instruct",
    "sc-15b": "bigcode/starcoder2-15b-instruct-v0.1",
    "ll-8b": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
}

DATASET = ["mce", "mcsn"]

LEVELS = ["method", "class", "repo"]

SYSTEM = "You're a specialized AI assisting with Python code summaries, deeply knowledgeable in computer science.\n"

METHOD_INSTRUCTION = "Concisely summarize the Python code provided in 1-3 sentences."

CLASS_CONTEXT_INSTRUCTION = (
    "Consider the following class code as additional context for your response:\n"
)
CLASS_INSTRUCTION = (
    "Concisely summarize the following Python function in 1-3 sentences:\n"
)

REPO_CONTEXT_INSTRUCTION = "You have the following repository context, which includes fragments of code with their corresponding paths and lines from the repository:\n\n"
REPO_INSTRUCTION_PREFIX = "Your task is to summarize the Python function located at "
REPO_INSTRUCTION_SUFFIX = (
    " concisely in 1-3 sentences, based on the provided context:\n"
)

GENERATION_CONFIGS = {}


def create_generation_configs(tokenizer):
    global GENERATION_CONFIGS
    GENERATION_CONFIGS = {
        "dsc": {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "do_sample": False,
            "top_k": 50,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "sc": {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "do_sample": False,
            "top_k": 50,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "eos_token_id": [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("###"),
            ],
            "pad_token_id": tokenizer.eos_token_id,
        },
        "ll": {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "do_sample": False,
            "top_k": 50,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "eos_token_id": [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        },
    }


def load_vector_stores(df, vector_stores_dir):
    def _load_vector_store(repo_name, embeddings):
        return FAISS.load_local(
            vector_stores_dir / repo_name.replace("/", "_"),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    repos = df["repo_name"].unique().tolist()
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda:2"},
        encode_kwargs={"normalize_embeddings": True},
    )
    global VECTOR_STORES
    VECTOR_STORES = {
        repo_name: _load_vector_store(repo_name, embeddings) for repo_name in repos
    }


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    df = pd.read_json(file_path, lines=True)
    return df


def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


# def retrieve_repo_context(method_code, repo_name, original_method_code, k):
#     context = VECTOR_STORES[repo_name].similarity_search(method_code, k=k)
#     return "\n\n".join(
#         [
#             f"File path: {d.metadata['file_path']}\nFile content:\n```\n{d.page_content}\n```"
#             for d in context
#             if d.page_content not in original_method_code
#         ]
#     )
def retrieve_repo_context(row, k):
    repo_name, method_code, original_method_code = row
    context = VECTOR_STORES[repo_name].similarity_search(method_code, k=k)
    return [d for d in context if d.page_content not in original_method_code]


def create_repo_context_prompt_content(context, k):
    return "\n\n".join(
        [
            f"File path: {d.metadata['file_path']}\nFile content:\n```\n{d.page_content}\n```"
            for d in context[:k]
        ]
    )


def method_level_pipeline(method_code, model, tokenizer, model_key, few_shots=[]):
    system_message = {
        "role": "system" if model_key != "sc" else "user",
        "content": SYSTEM,
    }
    main_message = {"role": "user", "content": f"{method_code}\n{METHOD_INSTRUCTION}"}
    messages = [system_message] + few_shots + [main_message]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        inputs,
        **GENERATION_CONFIGS[model_key],
    )
    return (
        tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
        .split("###")[0]
        .split("</s>")[0]
        .strip()
    )


def class_level_pipeline(method_code, class_context, model, tokenizer, model_key):
    system_message = {
        "role": "system" if model_key != "sc" else "user",
        "content": SYSTEM,
    }
    class_message = {
        "role": "user",
        "content": f"{CLASS_CONTEXT_INSTRUCTION}{class_context}",
    }
    main_message = {"role": "user", "content": f"{CLASS_INSTRUCTION}{method_code}"}
    messages = [system_message, class_message, main_message]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        inputs,
        **GENERATION_CONFIGS[model_key],
    )
    return (
        tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
        .split("###")[0]
        .split("</s>")[0]
        .strip()
    )


def repo_level_pipeline(row, model, tokenizer, model_key, few_shot_df, num_few_shot, k):
    method_code, method_path, repo_context, repo_name = row
    few_shots = construct_few_shot_list(
        few_shot_df, "repo", num=num_few_shot, repo_name=repo_name
    )
    system_message = {
        "role": "system" if model_key != "sc" else "user",
        "content": SYSTEM,
    }
    repo_message = (
        [
            {
                "role": "user",
                "content": f"{REPO_CONTEXT_INSTRUCTION}{create_repo_context_prompt_content(repo_context, k)}",
            }
        ]
        if k > 0
        else []
    )
    main_message = {
        "role": "user",
        "content": f"{REPO_INSTRUCTION_PREFIX}{method_path}{REPO_INSTRUCTION_SUFFIX}{method_code}",
    }
    messages = [system_message] + repo_message + few_shots + [main_message]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(
        inputs,
        **GENERATION_CONFIGS[model_key],
    )
    return (
        tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
        .split("###")[0]
        .split("</s>")[0]
        .strip()
    )


def create_predictions(
    df, model, tokenizer, level, model_key, few_shot_df, num_few_shot, k=0
):
    tqdm.pandas()
    if level == "method":
        few_shots = construct_few_shot_list(few_shot_df, level, num=num_few_shot)
        df["pred_summary"] = df["method_code"].progress_apply(
            lambda x: method_level_pipeline(x, model, tokenizer, model_key, few_shots)
        )
    elif level == "class":
        df["pred_summary"] = df.progress_apply(
            lambda x: class_level_pipeline(
                x.get("method_code"), x.get("class_code"), model, tokenizer, model_key
            ),
            axis=1,
        )
    elif level == "skeleton":
        df["pred_summary"] = df.progress_apply(
            lambda x: class_level_pipeline(
                x.get("method_code"), x.get("skeleton"), model, tokenizer, model_key
            ),
            axis=1,
        )
    elif level == "repo":
        df["pred_summary"] = df[
            ["method_code", "method_path", "repo_context", "repo_name"]
        ].progress_apply(
            lambda x: repo_level_pipeline(
                x, model, tokenizer, model_key, few_shot_df, num_few_shot, k
            ),
            axis=1,
        )
    return df


def save_predictions(df, model_dir, level, dataset, num_few_shot, k=None):
    df = df.reset_index()
    if level in ["class", "skeleton"]:
        output_file_path = model_dir / f"{level}-level-{dataset}-pred.jsonl"
    elif level == "method":
        output_file_path = (
            model_dir / f"{level}-level-{dataset}-few-shot-{num_few_shot}-pred.jsonl"
        )
    else:
        df.drop(["repo_context"], axis=1, inplace=True)
        output_file_path = (
            model_dir
            / f"{level}-level-{dataset}-few-shot-{num_few_shot}-context-{k}-pred.jsonl"
        )
    df.to_json(output_file_path, orient="records", lines=True)
    print(f"Predictions saved to {output_file_path}")


def construct_few_shot_list(df, level, num=0, repo_name=None):
    if repo_name is not None:
        df = df[df["repo_name"] == repo_name]
    df = df.sample(num, random_state=42)
    few_shot = []
    for code, summary in zip(df["method_code"], df["method_summary"]):
        if level == "repo":
            few_shot.append({"role": "user", "content": f"{CLASS_INSTRUCTION}\n{code}"})
        else:
            few_shot.append(
                {"role": "user", "content": f"{code}\n{METHOD_INSTRUCTION}"}
            )
        few_shot.append({"role": "assistant", "content": f"{summary}"})
    return few_shot


def inference(model_key, data_dir, dataset, level, num_few_shots):
    # Check if the data directory exists
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Directory {data_dir} does not exist.")

    PREPROCESSED_DATA_DIR = data_dir / "preprocessed"
    PREDICTED_DATA_DIR = data_dir / "predicted"
    VECTOR_STORES_DIR = data_dir / "vector-stores"

    dataset_file_path = PREPROCESSED_DATA_DIR / f"method-level-{dataset}.jsonl"
    if level != "class":
        dataset_few_shot_file_path = (
            PREPROCESSED_DATA_DIR / f"method-level-{dataset}-few-shot.jsonl"
        )
        few_shot_df = load_data(dataset_few_shot_file_path)
    df = load_data(dataset_file_path)

    model_name = model_names.get(model_key)
    if not model_name:
        raise ValueError(
            f"Invalid model key. Choose from: {', '.join(model_names.keys())}"
        )

    model, tokenizer = load_model_and_tokenizer(model_name)
    create_generation_configs(tokenizer)
    if level == "repo":
        print("Loading vector stores...")
        load_vector_stores(df, VECTOR_STORES_DIR)
        K = 50
        tqdm.pandas()
        df["repo_context"] = df[
            ["repo_name", "method_code", "original_method_code"]
        ].progress_apply(lambda x: retrieve_repo_context(x, K), axis=1)

    print(f"Model loaded: {model_name}")
    print(num_few_shots)
    for num_few_shot in num_few_shots:
        if dataset == "mce":
            print(f"Predicting with few shot examples: {num_few_shot} (level: {level})")
            df = create_predictions(
                df,
                model,
                tokenizer,
                level,
                model_key.split("-")[0],
                few_shot_df,
                num_few_shot,
            )
            MODEL_DIR = PREDICTED_DATA_DIR / model_name.split("/")[-1]
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            save_predictions(df, MODEL_DIR, level, dataset, num_few_shot)
            if level == "class":  # For skeleton (if level is class)
                print(
                    f"Predicting with few shot examples: {num_few_shot} (level: {level})"
                )
                df = create_predictions(
                    df,
                    model,
                    tokenizer,
                    "skeleton",
                    model_key.split("-")[0],
                    few_shot_df,
                    num_few_shot,
                )
                MODEL_DIR = PREDICTED_DATA_DIR / model_name.split("/")[-1]
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                save_predictions(df, MODEL_DIR, "skeleton", dataset, num_few_shot)
        else:  # mcsn
            if level == "repo":
                for k in [
                    12,
                    25,
                    50,
                ]:  # Number of documents to retrieve as a context for repo level
                    print(
                        f"Predicting with few shot examples: {num_few_shot} (level: {level} with {k} documents context)"
                    )
                    df = create_predictions(
                        df,
                        model,
                        tokenizer,
                        level,
                        model_key.split("-")[0],
                        few_shot_df,
                        num_few_shot,
                        k=k,
                    )
                    MODEL_DIR = PREDICTED_DATA_DIR / model_name.split("/")[-1]
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    save_predictions(df, MODEL_DIR, level, dataset, num_few_shot, k)
            else:  # method
                print(
                    f"Predicting with few shot examples: {num_few_shot} (level: {level})"
                )
                df = create_predictions(
                    df,
                    model,
                    tokenizer,
                    level,
                    model_key.split("-")[0],
                    few_shot_df,
                    num_few_shot,
                )
                MODEL_DIR = PREDICTED_DATA_DIR / model_name.split("/")[-1]
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                save_predictions(df, MODEL_DIR, level, dataset, num_few_shot)


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions using a PLM for Code Summarization task."
    )
    parser.add_argument("data_dir", type=Path, help="Directory with data")
    parser.add_argument("dataset", choices=DATASET, help="Dataset to use")
    parser.add_argument("level", choices=LEVELS, help="Level to use")
    parser.add_argument(
        "few_shots",
        type=str,
        help="List of number of few shot examples to use",
    )
    parser.add_argument(
        "model_key", choices=model_names.keys(), help="Key of the model from the list"
    )
    args = parser.parse_args()
    if args.few_shots == "all":
        num_few_shots = list(range(11))
    else:
        num_few_shots = [int(item) for item in args.few_shots.split(",")]
    inference(args.model_key, args.data_dir, args.dataset, args.level, num_few_shots)


if __name__ == "__main__":
    main()
