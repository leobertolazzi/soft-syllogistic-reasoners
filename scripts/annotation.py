import os
import yaml
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from dataset import get_dataset


ALL_SYLLO = ["AA1","AA2","AA3","AA4","AI1","AI2","AI3","AI4","AO1","AO2","AO3","AO4","AE1","AE2","AE3","AE4","IA1","IA2","IA3","IA4","II1","II2","II3","II4","IO1","IO2","IO3","IO4","IE1","IE2","IE3","IE4","OA1","OA2","OA3","OA4","OI1","OI2","OI3","OI4","OO1","OO2","OO3","OO4","OE1","OE2","OE3","OE4","EA1","EA2","EA3","EA4","EI1","EI2","EI3","EI4","EO1","EO2","EO3","EO4","EE1","EE2","EE3","EE4"]
VALID_SYLLO = ["AE1", "IA1", "AA1", "IE1", "EI1", "EA1", "EA2", "AI2", "AA2", "EI2", "IE2", "AE2", "EA3", "AE3", "AO3", "OA3", "IE3", "EI3", "AI4", "IA4", "AO4", "OA4", "AA4", "IE4", "EI4", "AE4", "EA4"]


def get_prompt(answer):
    return [
        {
            "role" : "system",
            "content" : """You are a language model designed to evaluate the truth of given statements based factual information about the world and the provided taxonomy. When presented with a sentence, analyze it and determine if it is true in the actual world. Provide a clear explanation for your conclusion, referencing factual information as necessary.

                ### Evaluation Guidelines:
                1. Consider common knowledge, widely accepted factual information, and the provided taxonomy.
                2. Assess the grammatical correctness of the sentence but focus on the factual accuracy.
                3. Provide a brief explanation supporting your evaluation, citing specific taxonomic relationships or factual information as needed.


                ### Taxonomy:
                * Siamese is a type of cat, which is a type of feline.
                * Labrador is a type of dog, which is a type of canine.
                * Angus is a type of cow, which is a type of mammal.
                * Chickadee is a type of bird, which is a type of winged animal.
                * Human is a type of animal, which is a type of mortal.
                * Sedan is a type of car, which is a type of vehicle.
                * Cruiser is a type of warship, which is a type of watercraft.
                * Boeing is a type of plane, which is a type of aircraft.
                * Daisy is a type of flower, which is a type of plant.
                * Pine is a type of evergreen, which is a type of tree.

                ### Examples:
                - Sentence: "All dogs are canines."
                Evaluation: True. According to the provided taxonomy, a Labrador is a type of dog, which is a type of canine. This relationship applies to all breeds of dogs, making the statement factually correct.

                - Sentence: "No human is mortal."
                Evaluation: False. According to the provided taxonomy, a Human is a type of animal, which is a type of mortal. This taxonomic relationship explicitly states that humans are mortals, making the statement incorrect.
                
                - Sentence: "Some cars are flowers."
                Evaluation: False. There is no taxonomic relationship between cars and flowers. Cars are a type of vehicle, while flowers are a type of plant. These two categories are distinct and unrelated, making the statement factually incorrect."""

        },
        {
            "role" : "user",
            "content" : f"Evaluate the following sentence and determine if it is true in the actual world:\n\n{answer}"
        }
    ]


def load_model(model_id, bnb_config):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=2000, cache_dir=CACHE_DIR, trust_remote_code=True, use_fast=True)
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=DEVICE, # dispatch efficiently the model on the available ressources
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Modify generation config
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    return model, tokenizer


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True
    )
    return bnb_config


def extract_options(dataset):
    to_annotate = []
    indexes = get_valid_syllogisms_indexes()
    texts, answers = [dataset["text"][i] for i in indexes], [dataset["answer"][i] for i in indexes]
    for text, answer in zip(texts, answers):
        options = text.split("Options:\n")[1].split("\n\nAnswer:")[0].split(".\n")
        options = [option.strip(".") for option in options]
        answer = answer.lower()
        for option in options:
            if option.lower() not in answer and option.lower() != "nothing follows":
                to_annotate.append(option)
    to_annotate = random.sample(to_annotate, len(to_annotate))
    return list(set(to_annotate))


def get_valid_syllogisms_indexes():
    # Syllo labels for each element in accuracies list
    all_syllo_x_10 = []
    for syllo in ALL_SYLLO:
        all_syllo_x_10 += [syllo]*10
    # List of indexes that belong to accuracies list & are in VALID_SYLLO
    indexes = []
    for valid_syllo in VALID_SYLLO:
        start_index = all_syllo_x_10.index(valid_syllo)
        indexes.extend(range(start_index, start_index + 10))
    return indexes


def main():
    # get options to annotate
    dataset = get_dataset(test_data="violate")[0]["test"]
    to_annotate = extract_options(dataset)
    # model and tokenizer
    bnb_config = create_bnb_config()
    model, tokenizer = load_model("meta-llama/Meta-Llama-3-8B-Instruct", bnb_config)
    
    model_responses = []
    for sentence in tqdm(to_annotate, leave=False):
        
        prompt = get_prompt(sentence)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
        output_ids = model.generate(model_input["input_ids"], do_sample=False, num_beams=1, max_new_tokens=5)[0]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        response = response.split("Evaluation")[-1]
        response = "True" if "True" in response else "False"
        model_responses.append((sentence,response))
    
    if not os.path.exists("data/annotation"):
        os.makedirs("data/annotation")
    with open("data/annotation/annotation.csv", "w") as o:
        o.write("sentence,model_answer\n")
        for sentence, response in model_responses:
            o.write(f"{sentence},{response}\n")



if __name__ == "__main__":

    # Read config
    with open("config.yaml", "r") as f:
        CFG = yaml.load(f, Loader=yaml.Loader)

    CACHE_DIR = CFG["cache_dir"]
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    main()