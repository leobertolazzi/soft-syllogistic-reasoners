import os
import yaml
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataset import get_dataset
from utils import prompt_template_chat, prompt_template, get_result_dictionary, save_results, clean_model_output

def main():
    args = parse_arguments()
    setup_environment(args)
    
    print_header(args)
    
    dataset = get_dataset(
        train_epochs=args.epochs,
        val_epochs=5 if args.eval_setting == "icl-in" else 1,
        test_data_type=args.test_data
    )
    
    icl_examples = 5 if "icl" in args.eval_setting else 0
    results = get_result_dictionary(dataset, args.model, args.test_data, args.eval_setting, icl_examples, args.run_id)
    
    icl_dataset = get_icl_dataset(args.eval_setting, dataset)

    model, tokenizer = load_model_and_tokenizer(args)
    
    metrics_dict, annotations_dict = evaluate_dataset(dataset["test"], model, tokenizer, icl_dataset, args)
    
    update_results(results, metrics_dict, annotations_dict)
    save_results(results)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pythia-1.4b",
        choices=["pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-1.4b-inst", "llama-2-7b", "llama-2-7b-chat", "llama-2-13b", "llama-2-13b-chat", "llama-3-8b", "llama-3-8b-chat", "llama-3-70b", "llama-3-70b-chat"])
    parser.add_argument("--test_data", type=str, default="believable",
        choices=["believable", "unbelievable", "2_premises", "3_premises", "4_premises"])
    parser.add_argument("--eval_setting", type=str, default="zero-shot-cot",
        choices=["zero-shot-cot", "icl-out", "icl-in", "sft"])
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--attention", type=str, default="eager",
        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--log", action=argparse.BooleanOptionalAction)
    parser.add_argument("--login", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def setup_environment(args):
    with open("config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    args.cache_dir = cfg["cache_dir"]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_id = cfg["models"][args.model]
    args.chat_model = "chat" in args.model
    args.epochs = 100
    args.in_context_examples = 5

    if "cot" in args.eval_setting:
        args.max_new_tokens = 70 if args.test_data in ["3_premises", "4_premises"] or "chat" in args.model else 50
        args.additional_prompt = " Let's think this through, step by step"
        args.final_answer_prompt = ". So, my final answer(s) is/are:"
    else:
        args.max_new_tokens = 20
        args.additional_prompt = " Given the premises I choose the following option(s): "

    args.quant_config = BitsAndBytesConfig(load_in_4bit=True) if "70" in args.model else \
                        BitsAndBytesConfig(load_in_8bit=True) if "13" in args.model else None

    os.environ['HF_HOME'] = "/mnt/cimec-storage6/shared/hf_llms_checkpoints"

    if args.login:
        huggingface_hub.login(cfg["hf_key"])

    if args.eval_setting == "sft":
        args.personal_dir = os.path.join(cfg["personal_dir"], "soft-syllogistic-reasoners", args.model)

    return args


def print_header(args):
    tqdm.write("-" * 50)
    tqdm.write(f"MODEL: {args.model}")
    tqdm.write(f"RUN ID: {args.run_id}")
    tqdm.write(f"TEST DATA TYPE: {args.test_data}")
    tqdm.write(f"EVAL PROCEDURE: {args.eval_setting}")
    tqdm.write("-" * 50)


def get_icl_dataset(eval_setting, dataset):
    if "icl-in" in eval_setting or "icl-out" in eval_setting:
        return dataset["val"]
    return None


def evaluate_dataset(dataset, model, tokenizer, icl_dataset, args):
    metrics_dict = defaultdict(list)
    annotations_dict = defaultdict(list)

    for sample in tqdm(dataset):
        problem, target, problem_type = sample["text"], sample["answer"], sample["type"]
        pred = generate(problem, target, problem_type, model, tokenizer, icl_dataset, args)
        answer_dict = eval_answer(pred, target)
        
        annotations_dict[problem_type].append(answer_dict["correct"])
        for key, value in answer_dict.items():
            metrics_dict[key].append(value)
    
    return metrics_dict, annotations_dict


def update_results(results, metrics_dict, annotations_dict):
    for key, value in metrics_dict.items():
        if key == "correct":
            results["results"]["accuracy"] = f"{np.mean(value)}"
            results["results"][key] = "|".join(map(str, value))
        else:
            results["results"][key] = f"{np.mean(value)}"

    for annotation, values in annotations_dict.items():
        results["results"][annotation] = sum(values) / len(values)


def generate(problem, target, problem_type, model, tokenizer, icl_dataset, args):
    context = get_context(problem_type, icl_dataset, args)
    template = get_template(context, problem, args, tokenizer)
    
    if "cot" in args.eval_setting:
        text = generate_cot(template, model, tokenizer, args)
        pred = generate_final_answer(text, target, model, tokenizer, args)
    else:
        pred = generate_direct(template, target, model, tokenizer, args)
    
    if args.log:
        log_generation(text if "cot" in args.eval_setting else template, pred, target)
    
    return pred


def get_context(problem_type, icl_dataset, args):
    if icl_dataset is None:
        return ""
    
    icl_df = icl_dataset.to_pandas()
    if "icl-in" == args.eval_setting:
        filtered_df = icl_df[icl_df["type"] == problem_type]
    else:
        filtered_df = icl_df[icl_df["type"] != problem_type]
    
    samples = filtered_df.sample(n=args.in_context_examples)
    return "\n\n".join([t + a for t, a in zip(samples["text"], samples["answer"])]) + "\n\n"


def get_template(context, problem, args, tokenizer):
    if args.chat_model:
        template = prompt_template_chat(context, problem, icl=context)
        return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
    return prompt_template(context, problem, icl=context)


def generate_cot(template, model, tokenizer, args):
    inputs = tokenizer(template + args.additional_prompt, return_tensors="pt").to(args.device)
    outputs = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens)
    return clean_model_output(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])


def generate_final_answer(cot_text, target, model, tokenizer, args):
    inputs = tokenizer(cot_text + args.final_answer_prompt, return_tensors="pt").to(args.device)
    final_answer_tokens = tokenizer(target, return_tensors="pt")["input_ids"].shape[1] + 2 if args.test_data == "avicenna" else 20
    outputs = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=final_answer_tokens)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split(args.final_answer_prompt)[-1]


def generate_direct(template, target, model, tokenizer, args):
    inputs = tokenizer(template + args.additional_prompt, return_tensors="pt").to(args.device)
    final_answer_tokens = tokenizer(target, return_tensors="pt")["input_ids"].shape[1] + 2 if args.test_data == "avicenna" else 20
    outputs = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=final_answer_tokens)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split(args.additional_prompt)[-1]


def log_generation(text, pred, target):
    tqdm.write(f"TEXT:\n{text}")
    tqdm.write(f"\n\nPREDICTION:\n{pred}")
    tqdm.write(f"\n\nTARGET:\n{target}")
    tqdm.write("-" * 50)


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, max_length=2000, cache_dir=args.cache_dir, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": args.attention,
        "cache_dir": args.cache_dir,
        "trust_remote_code": True
    }
    
    if args.eval_setting == "sft":
        adapter_path = os.path.join(args.personal_dir, f"{args.model}_run_{args.run_id}", f"{args.model}_run_{args.run_id}_best")
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        model.load_adapter(adapter_path)
    else:
        if args.quant_config:
            model_kwargs["quantization_config"] = args.quant_config
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()
    
    return model, tokenizer


def eval_answer(prediction, target):
    answers = {"correct": 0}
    target_values = target.lower().strip(".").split(" or ")
    prediction_lower = prediction.lower()
    
    if any(t in prediction_lower for t in target_values):
        answers["correct"] = 1
    
    return answers


if __name__ == "__main__":
    
    main()
