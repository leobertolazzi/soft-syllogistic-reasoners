import os
import random
import json
import pandas as pd
from typing import List, Dict, Union
from dataclasses import dataclass
from itertools import product
from datasets import Dataset
from utils import capfirst, lowerfirst


@dataclass
class Config:
    train_epochs: int = 100
    val_epochs: int = 1
    test_data_type: str = "believable"
    schemes_dir: str = "data/schemes"
    vocab_dir: str = "data/vocabulary"
    test_dir: str = "data/test"
    valid_test_data: List[str] = (
        "believable",
        "unbelievable",
        "2_premises",
        "3_premises",
        "4_premises",
    )


class SyllogismProcessor:
    def __init__(self, config: Config):
        self.config = config
        self._create_directories()
        self._initialize_constants()
        self._validate_config()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.config.schemes_dir, self.config.vocab_dir, self.config.test_dir]:
            os.makedirs(directory, exist_ok=True)

    def _initialize_constants(self) -> None:
        """Initialize constant values used throughout the process."""
        self.quantifiers = {
            "len_1": {
                "A": "All X are Y",
                "I": "Some X are Y",
                "O": "Some X are not Y",
                "E": "No X are Y"
            },
            "len_2": {
                "A": "All X are Z. All Z are Y",
                "I": "Some X are Y",
                "O": "Some X are not Y",
                "E": "No X are Y"
            },
            "len_3": {
                "A": "All X are Z. All Z are W. All W are Y",
                "I": "Some X are Y",
                "O": "Some X are not Y",
                "E": "No X are Y"
            }
        }

        self.figures = {
            "1": [("A", "B"), ("B", "C")],
            "2": [("B", "A"), ("C", "B")],
            "3": [("A", "B"), ("C", "B")],
            "4": [("B", "A"), ("B", "C")],
        }

        self.valid_syllogisms = {
            "AA1" : ["All A are C", "some A are C", "some C are A"],
            "IA1" : ["Some A are C", "some C are A"],
            "EA1" : ["Some C are not A"],
            "EI1" : ["Some C are not A"],
            "AE1" : ["No A are C", "no C are A", "some A are not C", "some C are not A"],
            "IE1" : ["Some A are not C"],
            "EA2" : ["No A are C", "no C are A", "some A are not C", "some C are not A"],
            "AI2" : ["Some A are C", "some C are A"],
            "AA2" : ["All C are A", "some A are C", "some C are A"],
            "EI2" : ["Some C are not A"],
            "IE2" : ["Some A are not C"],
            "AE2" : ["Some A are not C"],
            "EA3" : ["No A are C", "no C are A", "some A are not C", "some C are not A"],
            "AE3" : ["No A are C", "no C are A", "some A are not C", "some C are not A"],
            "AO3" : ["Some C are not A"],
            "OA3" : ["Some A are not C"],
            "IE3" : ["Some A are not C"],
            "EI3" : ["Some C are not A"],
            "AI4" : ["Some A are C", "some C are A"], 
            "IA4" : ["Some A are C", "some C are A"],
            "AO4" : ["Some A are not C"],
            "OA4" : ["Some C are not A"],
            "AA4" : ["Some A are C", "some C are A"],
            "IE4" : ["Some A are not C"],
            "EI4" : ["Some C are not A"],
            "AE4" : ["Some A are not C"],
            "EA4" : ["Some C are not A"],
        }

        self.options = [
            "All A are C.",
            "Some A are C.",
            "No A are C.",
            "Some A are not C.",
            "All C are A.",
            "Some C are A.",
            "No C are A.",
            "Some C are not A.",
            "Nothing follows."
        ]

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.test_data_type not in self.config.valid_test_data:
            raise ValueError(
                f"Invalid test_data_type: {self.config.test_data_type}. "
                f"Must be one of: {', '.join(self.config.valid_test_data)}"
            )

    def generate_schemes(self, gibberish: bool = False, len_chain: int = 1) -> None:
        """Generate syllogism schemes and save them to files."""
        structures = []
        answers = []
        names = []

        permutations = list(product(self.quantifiers["len_1"].keys(), repeat=2))
        
        for quantifier in permutations:
            if not gibberish:
                for figure in self.figures.keys():
                    major_name, minor_name = quantifier
                    major_x, major_y = self.figures[figure][0]
                    minor_x, minor_y = self.figures[figure][1]
                    name = f"{major_name}{minor_name}{figure}"

                    # Generate premises of standard syllogisms
                    premise_1 = self.quantifiers["len_1"][major_name].replace("X", major_x).replace("Y", major_y)
                    premise_2 = self.quantifiers["len_1"][minor_name].replace("X", minor_x).replace("Y", minor_y)

                    structure = f"Premise 1: {premise_1} / Premise 2: {premise_2}"
                    answer = " | ".join(self.valid_syllogisms.get(name, ["Nothing follows"]))

                    structures.append(structure)
                    answers.append(answer)
                    names.append(name)

            elif "A" in quantifier:
                for figure in self.figures.keys():
                    major_name, minor_name = quantifier
                    major_x, major_y = self.figures[figure][0]
                    minor_x, minor_y = self.figures[figure][1]
                    name = f"{major_name}{minor_name}{figure}"

                    # Generate premises using gibberish and based on chain length
                    premise_1 = self.quantifiers[f"len_{len_chain}"][major_name].replace("X", major_x).replace("Y", major_y)
                    premise_2 = self.quantifiers[f"len_{len_chain}"][minor_name].replace("X", minor_x).replace("Y", minor_y)

                    structure = []
                    premise_a = premise_1 if len(premise_1) >= len(premise_2) else premise_2
                    pos_premise_a = 1 if premise_a == premise_1 else 2

                    if pos_premise_a == 2:
                        premise_1 = premise_1.replace("Z. All Z are W. All W are ", "").replace("Z. All Z are ", "")
                        structure.append(f"Premise 1: {premise_1}")

                    premise_a = premise_a.split(".")
                    for n, a in enumerate(premise_a):
                        structure.append(f"Premise {n+pos_premise_a}: {a}")

                    if pos_premise_a == 1:
                        premise_2 = premise_2.replace("Z. All Z are W. All W are ", "").replace("Z. All Z are ", "")
                        structure.append(f"Premise {len_chain+1}: {premise_2}")

                    structure = " / ".join(structure).replace("  ", " ")
                    answer = " | ".join(self.valid_syllogisms.get(name, ["Nothing follows"]))

                    structures.append(structure)
                    answers.append(answer)
                    names.append(name)

        # Save schemes to file
        filename = (f"schemes_syllogisms_{len_chain+1}.txt" if gibberish 
                   else "schemes_syllogisms.txt")
        path = os.path.join(self.config.schemes_dir, filename)
        
        with open(path, "w") as f:
            for name, structure, answer in zip(names, structures, answers):
                f.write(f"<ID> {name}\n<STRUCTURE> {structure}\n<CONCLUSION> {answer}\n####\n")

    def _schemes_exist(self) -> bool:
        """Check if all necessary scheme files exist."""
        basic_scheme = os.path.join(self.config.schemes_dir, "schemes_syllogisms.txt")
        if not os.path.exists(basic_scheme):
            return False

        if "premises" in self.config.test_data_type:
            chain_length = int(self.config.test_data_type[0])
            chain_scheme = os.path.join(self.config.schemes_dir, f"schemes_syllogisms_{chain_length}.txt")
            return os.path.exists(chain_scheme)

        return True

    def ensure_schemes_exist(self) -> None:
        """Generate schemes if they don't exist."""
        if not self._schemes_exist():
            print("Generating schemes...")
            # Generate standard schemes
            self.generate_schemes(gibberish=False)
            # Generate gibberish schemes if needed
            if "premises" in self.config.test_data_type:
                chain_length = int(self.config.test_data_type[0]) - 1
                self.generate_schemes(gibberish=True, len_chain=chain_length)
        else:
            print("Schemes already exist. Skipping generation.")


class SyllogismDataset(SyllogismProcessor):
    def __init__(self, config: Config):
        super().__init__(config)
        self.ensure_schemes_exist()
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        """Load the appropriate vocabulary based on test data type."""
        vocab_path = f"data/vocabulary/syllowords_{self.config.test_data_type}.json"
        with open(vocab_path) as f:
            self.syllowords = json.load(f)

    def _load_schemes(self, split: str) -> List[str]:
        """Load syllogism schemes based on split and test data type."""
        if "premises" in self.config.test_data_type and split == "test":
            schemes_path = f"data/schemes/schemes_syllogisms_{self.config.test_data_type[0]}.txt"
        else:
            schemes_path = "data/schemes/schemes_syllogisms.txt"
        with open(schemes_path) as f:
            return f.read().split("\n####\n")[:-1]

    def get_conclusions_order(self, answers: List[str]) -> List[Dict[int, str]]:
        """Determine the order of terms in conclusions."""
        conclusions_order = []
        for answer in answers:
            order_dict = {}
            answer = lowerfirst(answer)
            for idx, ans in enumerate(answer.split("or")):
                if "nothing" in ans:
                    order_dict[idx] = "none"
                elif ans.find("A") < ans.find("C"):
                    order_dict[idx] = "AC"
                elif ans.find("C") < ans.find("A"):
                    order_dict[idx] = "CA"
            conclusions_order.append(order_dict)
        return conclusions_order

    def _process_structure(self, structure: str, problem_type: str) -> List[tuple]:
        """Process a single structure into multiple examples."""
        results = []
        to_substitute = self.syllowords[problem_type]
        
        for _ in range(len(to_substitute)):
            options = random.sample(self.options, len(self.options))
            
            # Extract base text and answer
            txt = structure.split("\n<STRUCTURE> ")[1].split("\n<CONCLUSION>")[0].replace(" / ", ".\n") + "."
            txt = f"Syllogism:\n\n{txt}\n\nOptions:\n{chr(10).join(options)}\n\nAnswer: "
            raw = txt
            ans = structure.split("<CONCLUSION> ")[1].replace("|", "or") + "."
            
            # Substitute with original words for belivable and unbelievable syllogisms
            if "premises" not in self.config.test_data_type:
                noun_a, noun_b, noun_c = to_substitute[_]
                for old, new in [("A", noun_a), ("B", noun_b), ("C", noun_c)]:
                    txt = txt.replace(f"{old} ", f"{new} ").replace(f"{old}.", f"{new}.")
            
            results.append((raw, txt, ans))
        
        return results

    def create_data_df(self, split: str = "train") -> pd.DataFrame:
        """Create a DataFrame with syllogism data."""
        syllo_structures = self._load_schemes(split)
        
        # Generate problem types
        problem_types = []
        texts, structures, answers = [], [], []
        
        for structure in syllo_structures:
            problem_type = structure.split("\n<STRUCTURE>")[0].replace("<ID> ", "")
            processed_examples = self._process_structure(structure, problem_type)
            
            for raw, txt, ans in processed_examples:
                problem_types.append(problem_type)
                texts.append(txt)
                structures.append(raw)
                answers.append(ans)

        return pd.DataFrame({
            "text": texts,
            "structures": structures,
            "answers": answers,
            "type": problem_types,
            "term_order": self.get_conclusions_order(answers)
        })

    def substitute_words(self, df: pd.DataFrame, split: str = "train") -> List[Union[str, tuple]]:
        """Substitute words with gibberish vocabulary."""
        with open("data/vocabulary/gibberish.json") as f:
            words = json.load(f)[split]
            
        substituted_data = []
        len_df = len(df)
        
        for i in range(0, len_df, 10):
            structure = df["structures"].iloc[i]
            answer = df["answers"].iloc[i]
            typ = df["type"].iloc[i]
            
            modified_structure = structure.split("Answer: ")[0] + "Answer: "
            
            # Handle answer modification based on split
            if split != "test":
                answer_parts = random.sample(answer.split(" or "), len(answer.split(" or ")))
                modified_answer = " or ".join(lowerfirst(ans.strip().strip(".")) for ans in answer_parts) + "."
                modified_answer = capfirst(modified_answer)
            else:
                modified_answer = answer
                
            # Substitute tokens
            extracted = []
            for tok in ["A", "B", "C", "Z", "W"]:
                while (selected_word := random.choice(words)) in extracted:
                    continue
                extracted.append(selected_word)
                modified_structure = modified_structure.replace(f"{tok} ", f"{selected_word} ").replace(f"{tok}.", f"{selected_word}.")
                modified_answer = modified_answer.replace(f"{tok} ", f"{selected_word} ").replace(f"{tok}.", f"{selected_word}.")
            
            if split in ["val", "test"]:
                substituted_data.append((modified_structure, modified_answer, typ))
            else:
                substituted_data.append(modified_structure + modified_answer)
        
        # Shuffle training data
        if split == "train":
            random.shuffle(substituted_data)
            
        return substituted_data

    def _generate_split(self, split: str) -> Dataset:
        """Generate dataset for a specific split."""
        if split == "train":
            train_data = []
            for _ in range(self.config.train_epochs):
                df = self.create_data_df(split="train")
                train_data += self.substitute_words(df, split="train")
            return Dataset.from_dict({"text": train_data})
            
        elif split == "val":
            df = self.create_data_df(split="val")
            val_substituted = []
            for _ in range(self.config.val_epochs):
                val_substituted.extend(self.substitute_words(df, split="val"))
            return Dataset.from_dict({
                "text": [t for t, _, _ in val_substituted],
                "answer": [a for _, a, _ in val_substituted],
                "type": [t for _, _, t in val_substituted]
            })
            
        else:  # test split
            return self._generate_test_split()

    def _generate_test_split(self) -> Dataset:
        """Generate test dataset."""
        test_path = self._get_test_path()
        
        if not os.path.exists(test_path):
            self._create_test_file(test_path)
            
        test_data = pd.read_json(path_or_buf=test_path, lines=True)
        return Dataset.from_pandas(test_data)

    def _get_test_path(self) -> str:
        """Get the appropriate test file path."""
        return f"data/test/syllogisms_{self.config.test_data_type}.jsonl"

    def _create_test_file(self, path: str) -> None:
        """Create test file if it doesn't exist."""
        test_data = self.create_data_df(split="test")
        
        if "premises" in self.config.test_data_type:
            examples = []
            for _ in range(10):
                examples.extend(self.substitute_words(test_data, split="test"))
                
            types = [test_data["type"].iloc[i] for i in range(0, len(test_data), 10)] * 10
            term_orders = [test_data["term_order"].iloc[i] for i in range(0, len(test_data), 10)] * 10
            
            test_data = pd.DataFrame({
                "text": [ex[0] for ex in examples],
                "answer": [ex[1] for ex in examples],
                "type": types,
                "term_order": term_orders
            })
        else:
            inputs = [txt.split("Answer: ")[0] + "Answer: " for txt in test_data['text']]
            answers = [txt.split("Answer: ")[1] for txt in test_data['text']]
            test_data["text"] = inputs
            test_data["answer"] = answers
            test_data = test_data[['text', 'answer', 'type', 'term_order']]
            
        test_data.to_json(path, orient='records', lines=True)

    def get_dataset(self) -> Dict[str, Dataset]:
        """Generate complete dataset with all splits."""
        return {
            "train": self._generate_split("train"),
            "val": self._generate_split("val"),
            "test": self._generate_split("test")
        }


def get_dataset(
    train_epochs: int = 100,
    val_epochs: int = 1,
    test_data_type: str = "believable"
) -> Dict[str, Dataset]:
    """Convenience function to create and return a dataset."""
    config = Config(
        train_epochs=train_epochs,
        val_epochs=val_epochs,
        test_data_type=test_data_type
    )
    dataset = SyllogismDataset(config)
    return dataset.get_dataset()


if __name__ == "__main__":

    dataset = get_dataset(
        train_epochs=100,
        val_epochs=1,
        test_data_type="believable"
    )

    print("\nDataset generated successfully!")
    print("\nSample counts:")
    print(f"Train: {len(dataset['train'])}")
    print(f"Val: {len(dataset['val'])}")
    print(f"Test: {len(dataset['test'])}")

    print("\nSample from each split:")
    print("\nTRAINING SAMPLE:")
    print(dataset["train"]["text"][0])
    print("\nVAL SAMPLE:")
    print(dataset["val"]["text"][0])
    print("\nTEST SAMPLE:")
    print(dataset["test"]["text"][0])
