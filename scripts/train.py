import os
import yaml
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from tqdm import tqdm
from utils import log_training
from dataset import get_dataset


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pythia-1.4b",
        choices=["pythia-1.4b", "llama-3-8b"])
    parser.add_argument("--run_id", type=int, default=1)
    return parser.parse_args()


def setup_model(model_name, model_id, cache_dir):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if "pythia" not in model_name else None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return model


def setup_lora(model, lora_config):
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def preprocess_function(examples, tokenizer, sequence_length):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=sequence_length)


def setup_dataloader(dataset, tokenizer, batch_size, sequence_length):
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, sequence_length),
        batched=True,
        remove_columns=["text"]
    )
    tokenized_dataset.set_format("torch")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=batch_size)


@torch.inference_mode()
def generate(problem, target, model, tokenizer, device, max_new_tokens):
    inputs = tokenizer(problem, return_tensors="pt")["input_ids"].to(device)
    target_ids = tokenizer(target, return_tensors="pt")["input_ids"].to(device)
    losses = []

    for i in range(max_new_tokens):
        try:
            outputs = model(input_ids=inputs)
            predicted_ids = torch.argmax(outputs.logits.squeeze(0), dim=1)
            loss = torch.nn.functional.cross_entropy(outputs.logits.squeeze(0)[-1, :], target_ids[0, i])
            losses.append(loss.item())
            inputs = torch.cat([inputs, predicted_ids[-1].view(1,1)], dim=1)
        except IndexError:
            break

    text = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    predictions = text.split("Answer:")[-1].strip()
    average_loss = sum(losses) / len(losses)
    return predictions, average_loss


def eval_loop(val_data, model, tokenizer, device, max_new_tokens, ep_count):
    correct = []

    for sample in tqdm(val_data, desc="Validation", leave=False):
        problem, target = sample["text"], sample["answer"]
        pred, loss = generate(problem, target, model, tokenizer, device, max_new_tokens)

        if "or" in target:
            correct.append(any(tar.lower().strip(".") in pred.lower() for tar in target.split(" or ")))
        else:
            correct.append(target.lower().strip(".") in pred.lower())

    accuracy = sum(correct) / len(correct)
    log_training(SAVE_NAME, ep_count, accuracy)
    return accuracy


def train(model, tokenizer, train_dataloader, val_dataset, optimizer, device, cfg):
    model.train()
    max_train_steps = len(train_dataloader)
    save_dir = os.path.join(cfg["personal_dir"], "soft-syllogistic-reasoners", MODEL_NAME, SAVE_NAME)
    os.makedirs(save_dir, exist_ok=True)

    best_accuracy, best_loss, best_epoch = 0, float('inf'), 0
    len_epoch = max_train_steps // cfg["training"]["epochs"]

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / cfg["training"]["gradient_accumulation_steps"]
        loss.backward()

        if step % cfg["training"]["gradient_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % len_epoch == 0:
            ep_count = step // len_epoch
            val_accuracy = eval_loop(val_dataset, model, tokenizer, device, cfg["training"]["max_new_tokens"], ep_count)

            if val_accuracy >= best_accuracy:
                best_accuracy, best_epoch = val_accuracy, ep_count
                model.save_pretrained(os.path.join(save_dir, f"{SAVE_NAME}_best"))

            tqdm.write(f"epoch = {ep_count} | eval/acc = {val_accuracy:.4f}")

        if step % len_epoch * cfg["training"]["save_every_epoch"] == 0:
            ep_count = step // len_epoch
            model.save_pretrained(os.path.join(save_dir, f"{SAVE_NAME}_ep_{ep_count}"))

    tqdm.write(f"BEST MODEL: epoch = {best_epoch} | eval/acc = {best_accuracy:.4f}")


def main():
    args = setup_argparse()
    cfg = load_config()

    global MODEL_NAME, SAVE_NAME
    MODEL_NAME = args.model
    SAVE_NAME = f"{MODEL_NAME}_run_{args.run_id}"

    dataset = get_dataset(train_epochs=cfg["training"]["epochs"], val_epochs=1, test_data_type="standard")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["models"][MODEL_NAME],
        cache_dir=cfg["cache_dir"],
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = setup_model(MODEL_NAME, cfg["models"][MODEL_NAME], cfg["cache_dir"])
    
    lora_config = LoraConfig(
        r=cfg["training"]["lora"]["r"],
        lora_alpha=cfg["training"]["lora"]["lora_alpha"],
        lora_dropout=cfg["training"]["lora"]["lora_dropout"],
        bias=cfg["training"]["lora"]["bias"],
        target_modules=cfg["training"]["lora"]["target_modules"][MODEL_NAME].split(),
        task_type=cfg["training"]["lora"]["task_type"],
    )
    model = setup_lora(model, lora_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader = setup_dataloader(dataset["train"], tokenizer, cfg["training"]["micro_batch"], cfg["training"]["sequence_length"])

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])

    train(model, tokenizer, train_dataloader, dataset["val"], optimizer, device, cfg)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    
    main()
