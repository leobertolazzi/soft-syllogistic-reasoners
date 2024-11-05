from typing import List, Dict, Optional, Tuple
import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
from prettytable import PrettyTable


class ResultAnalysis:
  def __init__(self, results_dir : str):
    """
    Initialize the DataSetAnalysis class with a dataset.

    :param results_dir: The directory containing the results.
    """
    self.root_dir = results_dir

    self.syllogisms = {
      "all" : ["AA1","AA2","AA3","AA4","AI1","AI2","AI3","AI4","AO1","AO2","AO3","AO4","AE1","AE2","AE3","AE4","IA1","IA2","IA3","IA4","II1","II2","II3","II4","IO1","IO2","IO3","IO4","IE1","IE2","IE3","IE4","OA1","OA2","OA3","OA4","OI1","OI2","OI3","OI4","OO1","OO2","OO3","OO4","OE1","OE2","OE3","OE4","EA1","EA2","EA3","EA4","EI1","EI2","EI3","EI4","EO1","EO2","EO3","EO4","EE1","EE2","EE3","EE4"],
      "valid" : ["AE1", "IA1", "AA1", "IE1", "EI1", "EA1", "EA2", "AI2", "AA2", "EI2", "IE2", "AE2", "EA3", "AE3", "AO3", "OA3", "IE3", "EI3", "AI4", "IA4", "AO4", "OA4", "AA4", "IE4", "EI4", "AE4", "EA4"],
      "invalid" : ["AA3", "AI1", "AI3", "AO1", "AO2", "IA2", "IA3", "II2", "II3", "II4", "IO1", "IO2", "IO3", "IO4", "EE1", "EE2", "EE3", "EE4", "EO1", "EO2", "EO3", "EO4", "OA1", "OA2", "OI1", "OI2", "OI3", "OI4", "OE1", "OE2", "OE3", "OE4", "OO1", "OO2", "OO3", "OO4"]
    }

    # Predictions taken from ...
    self.heuristics_prediction = pd.DataFrame({
      'Syllogism': ['AA1', 'AA2', 'AA4', 'AI2', 'AI4', 'AE1', 'AE2', 'AE3', 'AE4', 'AO3', 'AO4', 'IA1', 'IA4', 'IE1', 'IE2', 'IE3', 'IE4', 'EA1', 'EA2', 'EA3', 'EA4', 'EI1', 'EI2', 'EI3', 'EI4', 'OA3', 'OA4', 'AA3', 'AI1', 'AI3', 'AO1', 'AO2', 'IA2', 'IA3', 'II1', 'II2', 'II3', 'II4', 'IO1', 'IO2', 'IO3', 'IO4', 'EE1', 'EE2', 'EE3', 'EE4', 'EO1', 'EO2', 'EO3', 'EO4', 'OA1', 'OA2', 'OI1', 'OI2', 'OI3', 'OI4', 'OE1', 'OE2', 'OE3', 'OE4', 'OO1', 'OO2', 'OO3', 'OO4'],
      'GT' : ['Aac, Iac, Ica', 'Aca, Iac, Ica', 'Iac, Ica', 'Iac, Ica', 'Iac, Ica', 'Eac, Eca, Oac, Oca', 'Oac', 'Eac, Eca, Oac, Oca', 'Oac', 'Oca', 'Oac', 'Iac, Ica', 'Iac, Ica', 'Oac', 'Oac', 'Oac', 'Oac', 'Oca', 'Eac, Eca, Oac, Oca', 'Eac, Eca, Oac, Oca', 'Oca', 'Oca', 'Oca', 'Oca', 'Oca', 'Oac', 'Oca', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC'],
      'Atmosphere': ['Aac, Aca', 'Aac, Aca', 'Aac, Aca', 'Iac, Ica', 'Iac, Ica', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Oac, Oca', 'Oac, Oca', 'Iac, Ica', 'Iac, Ica', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Aac, Aca', 'Iac, Ica', 'Iac, Ica', 'Oac, Oca', 'Oac, Oca', 'Iac, Ica', 'Iac, Ica', 'Iac, Ica', 'Iac, Ica', 'Iac, Ica', 'Iac, Ica', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca'],
      'Matching': ['Aac, Aca', 'Aac, Aca', 'Aac, Aca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Aac, Aca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca'],
      'Conversion': ['Aac, Aca', 'Aac, Aca', 'Aac, Aca', 'Iac, Ica', 'Iac, Ica', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Eac, Eca', 'Oac, Oca', 'Oac, Oca', 'NVC', 'NVC', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'Oac, Oca', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'Aac, Aca', 'Iac, Ica', 'Iac, Ica', 'Oac, Oca', 'Oac, Oca', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC', 'NVC'],
      'PHM': ['Aac, Aca, Iac, Ica', 'Aac, Aca, Iac, Ica', 'Aac, Aca, Iac, Ica', 'Ica, Oca', 'Ica, Oac', 'Eac, Oac', 'Eca, Oca', 'Eca, Oca', 'Eac, Oac', 'Oca, Ica', 'Oac, Iac', 'Iac, Oac', 'Ica, Oca', 'Eac, Oca', 'Eca, Oca', 'Eca, Oca', 'Eac, Oac', 'Eac, Oac', 'Eca, Oca', 'Eac, Oac', 'Eac, Oca', 'Eac, Oac', 'Eac, Oca', 'Eac, Oac', 'Eca, Oca', 'Oac, Iac', 'Oca, Ica', 'Aac, Aca, Iac, Ica', 'Iac, Oac', 'Ica, Oca', 'Oac, Iac', 'Oca, Ica', 'Ica, Oca', 'Iac, Oac', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Iac, Ica, Oac, Oca', 'Oac, Iac', 'Oca, Ica', 'Oca, Ica', 'Oac, Iac', 'Eac, Eca, Oac, Oca', 'Eac, Eca, Oac, Oca', 'Eac, Eca, Oac, Oca', 'Eac, Eca, Oac, Oca', 'Oac, Iac', 'Oca, Ica', 'Oca, Ica', 'Oac, Iac', 'Oac, Iac', 'Oca, Ica', 'Oca, Iac', 'Oca, Ica', 'Oac, Iac', 'Oca, Ica', 'Oca, Iac', 'Oca, Ica', 'Oca, Iac', 'Oca, Ica', 'Oac, Oca, Iac, Ica', 'Oac, Oca, Iac, Ica', 'Oac, Oca, Iac, Ica', 'Oac, Oca, Iac, Ica'],
    })

    # Scores taken from ...
    self.human_scores = {
        "valid" : [
          ('AE1', 0.9),
          ('IA1', 0.87),
          ('AA1', 0.8),
          ('IE1', 0.3),
          ('EI1', 0.2),
          ('EA1', 0.1),
          ('EA2', 0.9),
          ('AI2', 0.8),
          ('AA2', 0.65),
          ('EI2', 0.45),
          ('IE2', 0.1),
          ('AE2', 0.1),
          ('EA3', 0.9),
          ('AE3', 0.87),
          ('AO3', 0.4),
          ('OA3', 0.38),
          ('IE3', 0.35),
          ('EI3', 0.35),
          ('AI4', 0.8), 
          ('IA4', 0.78),
          ('AO4', 0.6),
          ('OA4', 0.56),
          ('AA4', 0.37),
          ('IE4', 0.35),
          ('EI4', 0.35),
          ('AE4', 0.2),
          ('EA4', 0.1)
        ],
        "invalid" : [
          ('AA3', 0.31),
          ('AI1', 0.16),
          ('AI3', 0.37),
          ('AO1', 0.14),
          ('AO2', 0.17),
          ('IA2', 0.12),
          ('IA3', 0.28),
          ('II2', 0.33),
          ('II3', 0.3),
          ('II4', 0.51),
          ('IO1', 0.61),
          ('IO2', 0.33),
          ('IO3', 0.49),
          ('IO4', 0.53),
          ('EE1', 0.54),
          ('EE2', 0.44),
          ('EE3', 0.44),
          ('EE4', 0.76),
          ('EO1', 0.66),
          ('EO2', 0.28),
          ('EO3', 0.47),
          ('EO4', 0.49),
          ('OA1', 0.57),
          ('OA2', 0.2),
          ('OI1', 0.13),
          ('OI2', 0.36), 
          ('OI3', 0.31),
          ('OI4', 0.49),
          ('OE1', 0.47),
          ('OE2', 0.37),
          ('OE3', 0.51),
          ('OE4', 0.47),
          ('OO1', 0.49),
          ('OO2', 0.37),
          ('OO3', 0.42),
          ('OO4', 0.64)
        ]
    }

    self.patterns = {
        'A': r".*(all\s(?P<first>[a-zA-Z]+-*[a-zA-Z]*)\sare\s(?P<second>[a-zA-Z]+-*[a-zA-Z]*)).*",
        'I': r".*(some\s(?P<first>[a-zA-Z]+-*[a-zA-Z]*)\sare\s(?P<second>[a-zA-Z]+-*[a-zA-Z]*)).*",
        'O': r".*(some\s(?P<first>[a-zA-Z]+-*[a-zA-Z]*)\sare\snot\s(?P<second>[a-zA-Z]+-*[a-zA-Z]*)).*",
        'E': r".*(no\s(?P<first>[a-zA-Z]+-*[a-zA-Z]*)\sare\s(?P<second>[a-zA-Z]+-*[a-zA-Z]*)).*",
        'NVC': r".*(nothing follows).*"
    }
    
  def _print_df(self, df: pd.DataFrame):
    """
    Print a DataFrame in a pretty format.
    """
    # Convert DataFrame to a list of dictionaries
    table_data = df.to_dict(orient='records')
    # Create a PrettyTable instance
    table = PrettyTable()
    # Add columns
    table.field_names = df.columns
    # Add rows
    for row in table_data:
        table.add_row(row.values())
    print(table)

  def _print_df_with_separators(self, df: pd.DataFrame):
    """
    Print DataFrame with separators between different types in a pretty format.
    """
    # Group the DataFrame by type
    grouped = df.groupby('type')
    # Create a PrettyTable instance
    table = PrettyTable()
    # Add columns
    table.field_names = df.columns
    # Add rows with separators between types
    first_group = True
    for type_name, group in grouped:
        if not first_group:
            # Add a separator row (empty row with dashes)
            separator_row = ['-' * len(str(col)) for col in df.columns]
            table.add_row(separator_row)
        # Add rows for current type
        group_data = group.to_dict(orient='records')
        for row in group_data:
            table.add_row(row.values())
        first_group = False
    print(table)

  def _aggregate_runs(self, df: pd.DataFrame):
    """
    Aggregate the results of multiple runs.
    """
    grouped = df.groupby(['model', 'setting'])
    columns_to_aggregate = df.columns.difference(['run_id', 'model', 'setting', 'correct'])
    grouped_mean = grouped[columns_to_aggregate].mean()
    grouped_std = grouped[columns_to_aggregate].std()
    aggregate_df = grouped_mean.add_suffix('_mean').reset_index()
    aggregate_df = aggregate_df.merge(grouped_std.add_suffix('_std').reset_index(), on=['model', 'setting'])
    for stat in columns_to_aggregate:
        aggregate_df[stat] = aggregate_df.apply(
          lambda row: f"{round((row[f'{stat}_mean'] * 100), 2)}" 
          if pd.isna(row[f'{stat}_std']) 
          else f"{round((row[f'{stat}_mean'] * 100), 2)}+-{round((row[f'{stat}_std'] * 100), 2)}", 
          axis=1
        )
        #aggregate_df[stat] = (aggregate_df[f'{stat}_mean'] * 100).round(2).astype(str) +'+-'+ (aggregate_df[f'{stat}_std'] * 100).round(2).astype(str)
    to_drop = [i + "_mean" for i in columns_to_aggregate] + [i + "_std" for i in columns_to_aggregate]
    aggregate_df.drop(columns=to_drop, inplace=True)
    return aggregate_df 
  
  def _best_run(self, df: pd.DataFrame):
    """
    Find the best run for each model and setting.
    """
    grouped = df.groupby(['model', 'setting'])
    max_accuracy_rows = grouped['accuracy'].idxmax()
    df = df.loc[max_accuracy_rows]
    return df
  
  def _get_model_predictions(self, path: str):
    """
    Extract model predictions from file.
    """
    with open(os.path.join("results", "qualitative", path)) as f:
        txt = f.read()
        answers = [i.split("PREDICTION:")[1].split("TARGET:")[0].strip("\n") for i in txt.split("-"*80)[2:-1]]
    return answers

  def _get_target_conclusions(self, path: str):
    """
    Extract target conclusions from file.
    """
    with open(os.path.join("results", "qualitative", path)) as f:
        txt = f.read()
        targets = [i.split("PREDICTION:")[1].split("TARGET:")[1].strip("\n") for i in txt.split("-"*80)[2:-1]]
    return targets
  
  def _preprocess_answer(
        self,
        answer: str,
    ) -> str:
    """
    Preprocess an answer for analysis.
    """
    answer = answer.lower().strip(".")
    answer = answer.replace("\n", "")
    return answer
  
  def _extract_answers(
        self,
        answer: str,
        convert_to_codes: bool = False,
        term_order: dict = None,
        allow_nvc: bool = True,
        first_match_only: bool = False,
    ) -> List[str]:
    """
    Extract categorical statements from a text string with configurable behavior.
    """
    # Preprocessing
    answer = self._preprocess_answer(answer)
    
    # Update patterns to allow or disallow NVC
    patterns = self.patterns
    patterns["NVC"] = r".*(nothing follows).*" if allow_nvc else None
    
    # Process each pattern and find matches
    matches = []
    for statement_type, pattern in patterns.items():
        if pattern is None:
            continue

        match = re.match(pattern, answer)
        if match:
            # Skip if it's an I statement with "not" (handled by O pattern)
            if statement_type == 'I' and 'not' in answer:
                continue
            # Skip if it's an E statement with "not"
            if statement_type == 'E' and 'not' in match.groups():
                continue
                
            extracted = match.group(1)
            if convert_to_codes and term_order:
                if statement_type == 'NVC':
                    result = statement_type
                else:
                    first_term = term_order.get(match.group('first'), 'n').lower()
                    second_term = term_order.get(match.group('second'), 'n').lower()
                    result = f"{statement_type}{first_term}{second_term}"
            else:
                result = extracted
            matches.append(result)

    # Return the first match
    if first_match_only:
        first_index = len(answer)
        first = "empty"
        if matches:
            for match in matches:
                if answer.find(match) < first_index:
                    first = match
                    first_index = answer.find(match)
        return [first] 
    
    else:
        return matches

  def _check_incoherence(
        self,
        answers: List[str],
    ) -> Tuple[float, float]:
    """
    Analyzes answers for logical incoherence using coded statement representation.
    """
    def is_incoherent(answer: str) -> bool:
        """
        Check if a set of coded statements is logically incoherent.
        Handles statements in format like 'Aab', 'Eab', 'Iab', 'Oab' and 'NVC' where:
        A = Universal affirmative (All A are B)
        E = Universal negative (No A are B)
        I = Particular affirmative (Some A are B)
        O = Particular negative (Some A are not B)
        NVC = No Valid Conclusion
        """
        
        if not answer:
            return False
            
        # Check for NVC contradictions
        has_nvc = "nothing follows" in answer
        has_categorical = np.sum([1 for mood in ["A", "E", "O"] if re.match(self.patterns[mood], answer)])
        has_categorical += 1 if (re.match(self.patterns["I"], answer) and not re.match(self.patterns["O"], answer)) else 0
        if has_nvc and has_categorical:
            return True
        
        # Check for other contradictions
        for mood in ["A", "E"]:
            match = re.match(self.patterns[mood], answer)
            if match:
                if mood == "A":
                    if re.match(rf".*some\s{match.group('first')}\sare\snot\s{match.group('second')}.*", answer):
                        return True
                else:
                    if re.match(self.patterns["I"], answer) and not re.match(self.patterns["O"], answer):
                        return True
        return False

    incoherent_count = 0
    incoherent_nvc_count = 0
    total_nvc = 0

    for answer in answers:
    
        answer = self._preprocess_answer(answer)

        # Check for incoherence
        if is_incoherent(answer):
            incoherent_count += 1
            
        # Track NVC-specific statistics
        if "nothing" in answer:
            total_nvc += 1
            if is_incoherent(answer):
                incoherent_nvc_count += 1

    # Calculate percentages with zero-division protection
    total_answers = len(answers)
    total_incoherence = round((incoherent_count / total_answers) * 100, 2) if total_answers else 0
    nvc_incoherence = round((incoherent_nvc_count / total_nvc) * 100, 2) if total_nvc else 0

    return total_incoherence, nvc_incoherence

  def _check_incompleteness(
        self,
        answers: List[str],
    ) -> Tuple[float, float, float]:
    """
    Analyzes answers for completeness by checking if all required statements are present.
    """

    complete_e = 0
    total_e = 0
    complete_i = 0
    total_i = 0

    for answer in answers:

        answer = self._preprocess_answer(answer)
        answer = answer.split(" or ") if " or " in answer else answer.split(".")
        
        # Count E and I statements
        e_count = sum([1 for ans in answer if re.match(self.patterns["E"], ans)])
        i_count = sum([1 for ans in answer if re.match(self.patterns["I"], ans) and not re.match(self.patterns["O"], ans)])
         
        if i_count == 2:
            complete_i += 1
        if i_count > 0:
            total_i += 1
            
        if e_count == 2:
            complete_e += 1
        if e_count > 0:
            total_e += 1

    overall_incompleteness = round(100 - (complete_e + complete_i) / (total_e + total_i) * 100, 2)
    e_incompleteness = round(100 - (complete_e / total_e) * 100, 2)
    i_incompleteness = round(100 - (complete_i / total_i) * 100, 2)

    return overall_incompleteness, e_incompleteness, i_incompleteness
  
  def _coherent_vs_complete_plot(
        self,
        model: List[str], 
        incoherent: List[float], 
        incomplete: List[float], 
        output_file: str ='results/images/incoherent_vs_incomplete.png'
    ) -> None:
    """
    Create a scatter plot comparing incoherent and incomplete scores for each model.
    """
    
    # combined score for better visualization
    combined_score = [((inc**2 + inco**2)**0.5) for inc, inco in zip(incomplete, incoherent)]
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(incoherent, incomplete, s=100, alpha=1, c=combined_score, cmap='viridis')

    # Add labels for each point
    for i, txt in enumerate(model):
      plt.annotate(txt, (incoherent[i], incomplete[i]), xytext=(5, 0), textcoords='offset points')

    # Set the title and labels
    plt.title('Model Performance: Inconsistent vs Incomplete', fontsize=16)
    plt.xlabel('Inconsistent (%)', fontsize=12)
    plt.ylabel('Incomplete (%)', fontsize=12)

    # Set the axes to go from 0 to 100
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a colorbar to show the combined score (distance from origin)
    scatter.set_array(combined_score)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Combined Score (distance from origin)', rotation=270, labelpad=20)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")

  def _overlap_with_heuristics(self, answers: List[str], split: str = "all", overlap: str = "mistakes") -> Tuple[np.array, List[int], List[str], float]:
    """
    Measure the overlap between the model predictions and the heuristics predictions.
    """
    overlaps_dict = defaultdict(list)
    total_ans = 0
    total_to_consider = 0
    
    for answer, problem_type in answers:
        if problem_type in self.syllogisms[split]:
            if overlap == "mistakes":
                ans_to_consider = [ans for ans in answer if ans not in self.heuristics_prediction[self.heuristics_prediction["Syllogism"] == problem_type]["GT"].item()]
            elif overlap == "correct":
                ans_to_consider = [ans for ans in answer if ans in self.heuristics_prediction[self.heuristics_prediction["Syllogism"] == problem_type]["GT"].item()]
            else:
                raise ValueError("Invalid overlap, choose between 'mistakes' and 'correct'")

            total_to_consider += len(ans_to_consider)
            total_ans += len(answer)
            if ans_to_consider:
                for ans in ans_to_consider:
                    for heur in ["Atmosphere", "Matching", "Conversion", "PHM"]:
                        if ans in self.heuristics_prediction[self.heuristics_prediction["Syllogism"] == problem_type][heur].item():
                            overlaps_dict[heur].append(1)
                        else:
                            overlaps_dict[heur].append(0)
          
    if total_to_consider == 0:
        overlaps = np.array([0., 0., 0., 0.])
    else:
        overlaps = np.array([np.mean(value) for key, value in overlaps_dict.items()])
    n_samples = [len(value) for key, value in overlaps_dict.items()]
    heuristics = [key for key, value in overlaps_dict.items()]
    proportion = total_to_consider/total_ans

    return overlaps, n_samples, heuristics, proportion

  
  def accuracy(self, compare_strategies: bool = True):
    """
    Compute accuracy on the test data sets and combine results into two dataframes:
    1. Believability results (believable vs unbelievable)
    2. Complexity results (2, 3, and 4 premises)
    """
    # Dictionaries to store results for each category
    believability_results = {}
    complexity_results = {}
    
    # Process all datasets
    for test_data in ["believable", "unbelievable", "2_premises", "3_premises", "4_premises"]:
        all_syllo = pd.read_csv(f"results/{test_data}.csv")
        valid = all_syllo.copy()
        invalid = all_syllo.copy()
        
        # Store base information for this dataset
        temp_results = {
            "model": self._aggregate_runs(all_syllo)["model"].to_list(),
            "setting": self._aggregate_runs(all_syllo)["setting"].to_list()
        }
        
        # Process each split
        for df, split, metric in [(all_syllo, "all", "accuracy"), (valid, "valid", "valid"), (invalid, "invalid", "invalid")]:
            if "premises" in test_data:
                all_columns = df.columns.tolist()
                filtered_columns = [col for col in all_columns if col in self.syllogisms[split]]
                df[split] = df[filtered_columns].mean(axis=1)
            else:
                df[split] = df[self.syllogisms[split]].mean(axis=1)
            
            df = df[["model", "setting", metric]]
            df = self._aggregate_runs(df)
            temp_results[split] = df[metric].to_list()
        
        # Create temporary dataframe and filter based on compare_strategies
        temp_df = pd.DataFrame(temp_results)
        if compare_strategies:
          temp_df = temp_df[temp_df["model"].isin(["llama-3-8b", "pythia-1.4b"])]
        else:
          temp_df = temp_df[temp_df["setting"].str.contains('cot')]
        
        # Store results in appropriate dictionary with dataset name as key
        if test_data in ["believable", "unbelievable"]:
            believability_results[test_data] = temp_df
        else:
            complexity_results[test_data] = temp_df
  
    # Combine believability results
    if believability_results:
        combined_believability = pd.concat(
            [df.assign(type=name) for name, df in believability_results.items()],
            ignore_index=True
        )
        print("\nResults for Believability:")
        self._print_df_with_separators(combined_believability)
    
    # Combine complexity results
    if complexity_results:
        combined_complexity = pd.concat(
            [df.assign(type=name) for name, df in complexity_results.items()],
            ignore_index=True
        )
        print("\nResults for Premise Complexity:")
        self._print_df_with_separators(combined_complexity)

  def top_1_accuracy(self, test_data : str = "believable"):
    """
    Compute top 1 accuracy on the test data.
    """
    temp_df = pd.read_csv(f"results/{test_data}.csv")
    temp_df = temp_df[temp_df["model"].isin(["llama-3-8b", "pythia-1.4b"])]
    
    # Initialize results dictionary
    results = defaultdict(list)
    results["model"] = [temp_df["model"].unique()[0]]*4 + [temp_df["model"].unique()[1]]*4
    results["setting"] = list(temp_df["setting"].unique())*2
       
    # Compute accuracies for both believable and unbelievable datasets
    prob_types = pd.read_json(f"data/test/syllogisms_{test_data}.jsonl", lines=True)["type"]

    for model in temp_df["model"].unique():
        for setting in temp_df["setting"].unique():
            # Read data
            answers = self._get_model_predictions(f"{model}_{setting}_{test_data}.txt")
            targets = self._get_target_conclusions(f"{model}_{setting}_{test_data}.txt")
            # Compute accuracy
            accuracy = []
            accuracy_valid = []
            accuracy_invalid = []
            
            for answer, target, problem_type in zip(answers, targets, prob_types):
                answer = self._extract_answers(answer, first_match_only=True)[0]

                correct = 1 if answer.lower().strip(".") in target.lower() else 0
                
                accuracy.append(correct)
                if problem_type in self.syllogisms["valid"]:
                    accuracy_valid.append(correct)
                else:
                    accuracy_invalid.append(correct)
            
            acc, valid, invalid = round(np.mean(accuracy)*100, 2), round(np.mean(accuracy_valid)*100, 2), round(np.mean(accuracy_invalid)*100, 2)
            # Store results
            results["accuracy"].append(acc)
            results["valid"].append(valid)
            results["invalid"].append(invalid)
    
    # Create and format DataFrame
    df = pd.DataFrame(results)
    print(f"Top 1 Accuracy on {test_data.capitalize()}:")
    self._print_df(df)


  def correlation_with_humans(self, test_data : str = "believable", compare_strategies : bool = False):
    """
    Compute and return the correlation matrix of the dataset.
    """
    results = pd.read_csv(f"results/{test_data}.csv")
    # valid syllogisms
    valid_models = results[[i[0] for i in self.human_scores["valid"]]]
    valid_corr = []
    for idx, row in valid_models.iterrows():
        corr = stats.spearmanr(row.to_numpy(), [i[1] for i in self.human_scores["valid"]], axis=1)[0]
        corr = corr.round(2)
        valid_corr.append(corr)
    # add to result dataframe
    results["valid_corr"] = valid_corr
    results = results[["model", "setting", "valid_corr"]]
    if compare_strategies:
        results = results[results["model"].isin(["llama-3-8b", "pythia-1.4b"])]
    else:
        results = results[results["setting"].str.contains('cot')]

    results = self._aggregate_runs(results)
    print("Correlation with human scores for valid syllogisms:")
    self._print_df(results)
    
  def consistency_and_completeness(
        self,
        models: List[str] = ["llama-3-8b", "pythia-1.4b"],
        settings: List[str] = ["zero-shot-cot", "icl-out", "icl-in", "sft"],
        test_set: str = "believable",
        analysis: str = "coherent_vs_complete",
    ) -> Optional[pd.DataFrame]:
    """
    Comprehensive syllogism analysis function that handles coherence and completeness checks.
    """
    if analysis == "consistent":
        data = {
            "model": [], 
            "setting": [], 
            "incoherent(c)": [], 
            "incoherent(c)(nvc)": []
        }
        
        for model in models:
            for setting in settings:
                
                filename = f"{model}_{setting}_{test_set}.txt"
                answers = self._get_model_predictions(filename)
                incoherent_c, incoherent_nvc = self._check_incoherence(answers)
                # Adjust NVC percentage based on overall incoherence
                incoherent_nvc = round((incoherent_nvc * (incoherent_c/100)), 2)
                
                data["model"].append(model)
                data["setting"].append(setting)
                data["incoherent(c)"].append(incoherent_c)
                data["incoherent(c)(nvc)"].append(incoherent_nvc)
                
        df = pd.DataFrame(data)
        print(f"Coherence on {test_set.capitalize()}")
        self._print_df(df)  # Assumes print_df function exists
        return df
        
    elif analysis == "complete":
        data = {
            "model": [], 
            "setting": [], 
            "incomplete": [], 
            "incomplete(I)": [], 
            "incomplete(E)": []
        }

        for model in models:
            for setting in settings:
                filename = f"{model}_{setting}_{test_set}.txt"
                answers = self._get_model_predictions(filename)
                incomplete, incomplete_e, incomplete_i = self._check_incompleteness(
                    answers
                )
                
                data["model"].append(model)
                data["setting"].append(setting)
                data["incomplete"].append(incomplete)
                data["incomplete(I)"].append(incomplete_i)
                data["incomplete(E)"].append(incomplete_e)
                
        df = pd.DataFrame(data)
        print(f"Completeness on {test_set.capitalize()}")
        self._print_df(df)
        return df
        
    elif analysis == "coherent_vs_complete":
        os.makedirs("results/images", exist_ok=True)
        model_list = []
        incoherent_results = []
        incomplete_results = []

        for model in models:
            for setting in settings:
                filename = f"{model}_{setting}_{test_set}.txt"
                answers = self._get_model_predictions(filename)
                
                incoherent_ans, _ = self._check_incoherence(answers)
                incomplete_ans, _, _ = self._check_incompleteness(answers)
                
                model_list.append(model)
                incoherent_results.append(incoherent_ans)
                incomplete_results.append(incomplete_ans)
        
        self._coherent_vs_complete_plot(model_list, incoherent_results, incomplete_results)
        return None

  def heuristics_overlap(self, test_data : str = "believable"):
    """
    Compute the overlap between model predictions and heuristics for a set of syllogisms
    """
    os.makedirs("results/images", exist_ok=True)
    df = pd.read_json(f"data/test/syllogisms_{test_data}.jsonl", lines=True)
    term_orders = df["term_order"].to_list()
    problem_type = df["type"].to_list()

    for split in ["valid", "invalid"]:
        print("----")
        # Overlap with heuristics for GT answers
        print(f"Proportion of GT answers predicted by heuristics on {split} syllogisms :\n")
        answers = []
        for ans, typ in zip(self.heuristics_prediction["GT"], self.heuristics_prediction["Syllogism"]):
            answers.append([ans.split(", "), typ])
        overlaps, n_samples, heuristics, error_rate = self._overlap_with_heuristics(answers, split=split, overlap="correct")
        for key, value in zip(heuristics, overlaps):
            print(f"{key}: {value*100}")
        
        print(f"\nProportion of model answers predicted by heuristics on {split} syllogisms :\n")
        # Overlap with heuristics for model answers
        for overlap in ["mistakes", "correct"]:
            
            results = []
            sample_sizes = []

            for model in ["llama-3-8b", "pythia-1.4b"]:
                for setting in ["zero-shot-cot", "icl-out", "icl-in", "sft"]:
                    answers_raw = self._get_model_predictions(f"{model}_{setting}_{test_data}.txt")
                    answers_raw = [i.split("\n\nSyllogism")[0].replace("winged animals", "winged-animals") for i in answers_raw]
                    answers = []
                    for ans, terms, typ in zip(answers_raw, term_orders, problem_type):
                        answers.append((self._extract_answers(ans, term_order=terms, convert_to_codes=True), typ))

                    overlaps, n_samples, heuristics, error_rate = self._overlap_with_heuristics(answers, split=split, overlap=overlap)
                    results.append(overlaps)
                    sample_sizes.append(n_samples)

            y_labels = [f"{model} {setting}" for model in ["llama-3-8b", "pythia-1.4b"] for setting in ["zero-shot-cot", "icl-out", "icl-in", "sft"]]
            x_labels = heuristics

            results = np.array(results)
            
            # Create the heatmap
            sns.heatmap(results*100, fmt='.2f', vmin=5, vmax=75, annot=True, cmap='viridis', xticklabels=x_labels, yticklabels=y_labels)
            
            # Display the heatmap
            plt.gca().set_aspect('equal')
            plt.xticks(rotation=90)
            plt.title(f"{test_data.capitalize()} {split.capitalize()} {overlap.capitalize()}")
            plt.tight_layout()
            if os.path.exists(f"results/images/heuristics_{test_data}_{split}_{overlap}.png"):
                print(f"Plot already exists at results/images/heuristics_{test_data}_{split}_{overlap}.png")
            else:
                plt.savefig(f"results/images/heuristics_{test_data}_{split}_{overlap}.png", dpi=300)
                print(f"Plot saved as results/images/heuristics_{test_data}_{split}_{overlap}.png")
            plt.close()


if __name__ == "__main__":

    # Run the analysis
    analysis = ResultAnalysis("results")

    print("---", "Accuracy results", "---\n")
    print("Comparison of Zero-shot Chain-of-Tought models")
    analysis.accuracy(compare_strategies=False)
    print("Comparison of Pythia-1.4B and LLaMA-3 8B with different strategies")
    analysis.accuracy(compare_strategies=True)
    print("-"*80)
    print()

    print("---", "Correlation with human scores", "---\n")
    print("Comparison of Zero-shot Chain-of-Tought models")
    analysis.correlation_with_humans(compare_strategies=False)
    print("Comparison of Pythia-1.4B and LLaMA-3 8B with different strategies")
    analysis.correlation_with_humans(compare_strategies=True)
    print("-"*80)
    print()

    print("---", "Consistency and Completeness results", "---\n")
    analysis.consistency_and_completeness(analysis="consistent")
    analysis.consistency_and_completeness(analysis="complete")
    analysis.consistency_and_completeness(analysis="coherent_vs_complete")
    print("-"*80)
    print()

    print("---", "Heuristics overlap results", "---\n")
    analysis.heuristics_overlap()
    print("-"*80)
    print()

    print("---", "Top 1 Accuracy results", "---\n")
    analysis.top_1_accuracy(test_data="believable")
    analysis.top_1_accuracy(test_data="unbelievable")
    print("-"*80)

