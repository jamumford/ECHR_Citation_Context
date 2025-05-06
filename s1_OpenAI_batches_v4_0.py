"""
Version History:
v4_0 = Adapted to produce outputs suitable for citation analysis annotation project.
v3_6 = Now allows processing of Decision-type (in addition to Judgment) outcomes.
v3_5 = Extended random sampling to also cover case outcome class ("violation, "nonviolation,
    or "unresolved").
v3_4 = Extended to CHAMBER and COMMITTEE cases, including random sampling by date.
v3_3 = Corrected mistake in passage-finding for bullet passages and subsections.
v3_2 = Simplified prompt, including removing application numbers from output format, to save on tokens.
v3_1 = Extended to two other passage types (subsection, and bullet paragraph).
v3_0 = Converted to produce batches for a full chamber-type dataset for full LAW section passages.
v2_0 = Converted code to produce prompts with the task extended from just identifying the Article
    to also seek to categorise the citation type (postive, negative, neutral).
v1_2 = Added batch prompt for Experiment 2: added intersection of articles; and 
    Experiment 3: added most recently referenced Article.    
v1_1 = Adjusted to accommodate dictionary based input data.
v1_0 = Creates batches for OpenAI API for Experiment 1: citation passage only.
"""

from collections import defaultdict
from config import openai_key
import json
from openai import OpenAI
import os 
import random
import re
from tqdm import tqdm


def check_single_match(section_markers, section_pattern, section_type, full_text, citing_case_id):
    
    # Search for matches in the full text
    #print(full_text)
    section_matches = re.findall(section_pattern, full_text)
    
    # Ensure there is at most one match
    if len(section_matches) == 0:
        raise ValueError(f"No valid {section_type} marker found in the text for {citing_case_id}. Expected one of {section_markers}.")
    elif len(section_matches) > 1:
        raise ValueError(f"Multiple {section_type} markers found in the text for {citing_case_id}. Ensure only one of {section_markers} is present.")


def document_segmentation(citing_case_id, full_text, experiment_type, court_level):

    # Define the list of possible markers
    #print(f"Processing citing_case_id: {citing_case_id}")
    law_markers = ["THE LAW", "AS TO THE LAW", "TO THE LAW", "LAW", "COMPLAINTS AND THE LAW"]
    if court_level == "3_COMMITTEE":
        law_markers += ["THE COURT’S ASSESSMENT"]
    verdict_markers = ["FOR THESE REASONS, THE COURT UNANIMOUSLY", "FOR THESE REASONS, THE COURT UNANIMOUSLY,", "FOR THESE REASONS, THE COURT, UNANIMOUSLY", "FOR THESE REASONS, THE COURT, UNANIMOUSLY,", "FOR THESE REASONS THE COURT UNANIMOUSLY", "FOR THESE REASONS, THE COURT", "FOR THESE REASONS, THE COURT,", "FOR THESE REASONS, THE COURT:", "FOR THESE REASONS THE COURT"]
    if court_level in ["4_DECGRANDCHAMBER", "5_ADMISSIBILITY", "6_ADMISSIBILITYCOM"]:
        verdict_markers += ["FOR THESE REASONS, THE COURT, BY A MAJORITY", "FOR THESE REASONS, THE COURT, BY A MAJORITY,", "For these reasons, the Court,", "For these reasons, the Court", "For these reasons, the Court, unanimously,", "For these reasons, the Court, by a majority,", "For these reasons, the Court by a majority"]
    
    # Create a regex pattern to match any of the markers within the desired structure
    law_pattern = r"\r\n\s*(" + "|".join(re.escape(marker) for marker in law_markers) + r")\s*\r"
    verdict_pattern = r"\r\n\s*(" + "|".join(re.escape(marker) for marker in verdict_markers) + r")\s*\r"
    
    # Search for matches in the full text and ensure only one match exists
    check_single_match(law_markers, law_pattern, "LAW", full_text, citing_case_id)
    check_single_match(verdict_markers, verdict_pattern, "VERDICT", full_text, citing_case_id)
    
    # Search for start and end markers in the full text
    start_match = re.search(law_pattern, full_text)
    end_match = re.search(verdict_pattern, full_text)
    
    # Validate that start and end markers are found
    if not start_match:
        raise ValueError(f"No valid start marker found in the text for {citing_case_id}. Expected one of {possible_markers}.")
    if not end_match:
        raise ValueError(f"No valid verdict marker found in the text for {citing_case_id}. Expected one of {verdict_markers}.")

    # Extract positions of the markers
    start_index = start_match.end()  # End of the start marker
    end_index = end_match.start()   # Start of the verdict marker

    # Validate marker order
    if start_index >= end_index:
        raise ValueError(f"Start marker occurs after or overlaps with the verdict marker in the text for {citing_case_id}.")

    # Split the text into sections based on marker positions
    #facts_section = full_text[:start_index].strip() # NOTE THAT THIS IS NOT CORRECT AS WE NEED PATTERNS FOR IDENTIFYING THE FACTS SECTION
    law_section = full_text[start_index:end_index].strip()

    # Segmentation logic based on experiment type
    if experiment_type == "law_section":
        return {"law section": law_section}  # Return only the law section
        
    elif experiment_type == "bullet_passages":
        # Find all instances of bullet passages in the law section
        bullet_pattern = r"\r\n\s*(\d+)\."
        bullet_matches = [match.group(1) for match in re.finditer(bullet_pattern, law_section)]

        # Extract the bullet passages
        bullet_passages = {}
        for i, match in enumerate(bullet_matches):
            # Find the start of the current passage
            passage_start = law_section.find(f"\r\n{match}.")
    
            # Determine the start of the next passage, if it exists
            if i + 1 < len(bullet_matches):
                next_match = bullet_matches[i + 1]
                passage_end = law_section.find(f"\r\n{next_match}.", passage_start)
            else:
                passage_end = len(law_section)  # Last passage ends at the end of the law section
    
            # Extract the passage text
            bullet_passages[f"{match}."] = law_section[passage_start:passage_end].strip()
    
        return bullet_passages
    
    elif experiment_type == "subsections":
        # Original subsection pattern
        subsection_pattern = r"\r\n\s*([A-Z])\.\s*(.*?)\s*\r"
        subsection_matches = list(re.finditer(subsection_pattern, law_section))
    
        # Extract subsections with their content
        subsections = {}
        
        # Handle the content before the first subsection match
        if subsection_matches:
            first_match_start = subsection_matches[0].start()  # Start of the first subsection
            subsections["Introduction"] = law_section[:first_match_start].strip()  # Content before the first subsection
        else:
            # If no subsections are found, return the entire law_section as "Introduction"
            return {"Full Law Section": law_section.strip()}
    
        # Extract the content of each subsection
        for i, match in enumerate(subsection_matches):
            title = f"{match.group(1)}. {match.group(2)}".strip()
            passage_start = match.end()  # Start of the content for the current subsection
    
            # Find the start of the next subsection (if it exists)
            if i + 1 < len(subsection_matches):
                passage_end = subsection_matches[i + 1].start()  # Start of the next subsection
            else:
                passage_end = len(law_section)  # Last subsection goes till the end of the law section
    
            # Extract and clean the subsection content
            subsections[title] = law_section[passage_start:passage_end].strip()
    
        return subsections
        
    else:
        raise ValueError(f"Invalid experiment_type: {experiment_type}")


def get_distribution(json_objects):
    """
    Computes the distribution of values for 'article', 'violation', and 'nonviolation'
    across a list of JSON objects (each provided as a JSON string). Only articles that are
    present in the set of important_articles (i.e. "2" through "18") are counted.

    Parameters:
        json_objects (list): List of JSON strings.

    Returns:
        dict: A dictionary with keys 'violation', 'nonviolation', and 'unresolved'.
              Each key maps to another dictionary that counts the occurrences of each
              article (filtered by the important articles).
    """
    # Define important articles as a set of strings "2" to "18"
    important_articles = {str(i) for i in range(2, 19)}
    
    # Initialize the distribution dictionary
    distribution = {"violation": {}, "nonviolation": {}, "unresolved": {}}
    article_outcomes = {"violation": {}, "nonviolation": {}, "unresolved": {}}
    
    for line in json_objects:
        obj = json.loads(line)
        
        # Process the "article" field and keep only those in important_articles
        articles_str = obj.get("article", "")
        raised_articles = {art.strip() for art in articles_str.split(";") if art.strip() in important_articles}
        
        # For each of the outcome keys, filter and count only articles in important_articles
        for key in ["violation", "nonviolation"]:
            key_str = obj.get(key, "")
            outcome_articles = {art.strip() for art in key_str.split(";") if art.strip() in important_articles}
            for article in outcome_articles:
                article_outcomes[key].setdefault(article, []).append(obj)
                distribution[key][article] = distribution[key].get(article, 0) + 1
            # Remove articles that have been counted as either violation or nonviolation
            raised_articles -= outcome_articles
        
        # Any remaining articles from the raised articles are unresolved
        for article in raised_articles:
            article_outcomes["unresolved"].setdefault(article, []).append(obj)
            distribution["unresolved"][article] = distribution["unresolved"].get(article, 0) + 1
    
    return distribution, article_outcomes


def get_prompt(citing_name, passage):

    # Prompt template
    base_prompt = '''Prompt:
    You are a legal assistant. Analyse a passage from a European Court of Human Rights case document: {citing_name}.
    Task: Identify all ECHR case law citations and for each, determine:
    1. Paragraph containing the citation.
    2. Case name.
    3. Relevant Article(s) or Protocol(s) of the Convention.
    4. Context of citation: ['positive', 'negative', 'neutral'], as defined below:
       - 'positive': The outcome, facts or reasoning in the citation is being applied to {citing_name}.
       - 'negative': The outcome, facts or reasoning in the citation is being distinguished from that relevant to {citing_name}.
       - 'neutral': The citation is technical or supporting without directly testing for violation in {citing_name}.
    
    Output format (JSON):
    {{
        "citations": [
            {{"paragraph": "...", "case_name": "...", "articles": ["..."], "context_class": "..."}}
        ]
    }}
    Instructions:
    - "paragraph": Extract the paragraph number at the start of the citation’s paragraph (format: "n.", where "n" is an integer).
    - "case_name": Use the format "A v. B", where "A" is the applicant and "B" the respondent nation state. Both "A" and "B" can include multiple names and/or nation states, and could involve references to "Others" (e.g., "DUARTE AGOSTINHO AND OTHERS v. PORTUGAL AND 32 OTHERS"). The case name may be abridged within the passage, (e.g., "DUARTE AGOSTINHO AND OTHERS"), especially where the case has been cited earlier in the wider case document. If abridged or incomplete, provide the known part without guessing or inventing names.
    - "articles": Use 'a' for Article numbers (e.g., '6') and 'Pb-c' for Protocols (e.g., 'P4-2').
    - "context_class": Write one class from ['positive', 'negative', 'neutral'] in accordance with the task instructions for interpreting the citation context.
    Passage:
    {passage}
    '''

    # Combine base prompt with variations and provided arguments
    return base_prompt.format(
        citing_name=citing_name,
        passage=passage
    )


def systematic_sample(valid_entries, case_limit):
    # Define outcome classes
    outcome_classes = ["violation", "nonviolation", "unresolved"]
    case_limit /= len(outcome_classes)
    
    # Get distribution and outcomes dictionaries.
    # article_distribution: { outcome_class: { article: count, ... }, ... }
    # article_outcomes: { outcome_class: { article: [obj1, obj2, ...], ... }, ... }
    article_distribution, article_outcomes = get_distribution(valid_entries)
    
    # Check distribution and outcome classes match expected outcome_classes
    if set(article_distribution.keys()) != set(outcome_classes):
        raise ValueError(f"article_distribution.keys(): {article_distribution.keys()} do not match outcome_classes: {outcome_classes}")
    if set(article_outcomes.keys()) != set(outcome_classes):
        raise ValueError(f"article_outcomes.keys(): {article_outcomes.keys()} do not match outcome_classes: {outcome_classes}")
    
    # Get ranking of Articles and Protocols by distribution
    article_ranks = {}
    for outcome_class, outcome_dict in article_distribution.items():
        article_ranks[outcome_class] = rank_keys_by_value(outcome_dict)

    # Prepare a nested dictionary to store the selected lines.
    # Keys will be the outcome_classes, and values will be lists of selected objects.
    selected_lines = {oc: [] for oc in outcome_classes}
    
    # For each outcome class, perform a round-robin selection:
    for outcome_class in outcome_classes:
        count = 0
        # Continue looping until we have collected case_limit objects for this outcome class.
        while count < case_limit:
            progress_made = False  # To track if any object is selected in this full pass.
            # Iterate through articles in the order determined by article_ranks.
            for article in article_ranks[outcome_class]:
                if count >= case_limit:
                    break  # Stop if we've reached the case limit.
                # Check if there are any remaining objects for this article.
                if article_outcomes[outcome_class][article]:
                    # Pop the first available object.
                    obj = article_outcomes[outcome_class][article].pop(0)
                    selected_lines[outcome_class].append(obj)
                    count += 1
                    progress_made = True
            # If no object was added during a complete pass, exit to avoid an infinite loop.
            if not progress_made:
                break
    
    return selected_lines


def rank_keys_by_value(d):
    # sorted returns a list of keys sorted by their values in descending order
    return sorted(d.keys(), key=lambda k: d[k], reverse=True)


def main(input_json_file, AI_model, court_level, experiment_type, task_type, case_limit):
    # Output file  
    output_dir = os.path.join('Batches', task_type, experiment_type, court_level)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{AI_model}.jsonl')
    
    # Setting maximum token output based on experiment_type
    token_dict = {"law_section": 15000, "subsections": 5000, "bullet_passages": 600}
    
    print(f"\nProcssing file: {output_file}")
    
    # Open the output file in append mode
    with open(output_file, "w", encoding="utf-8") as outfile:
        for outcome_cls, line in tqdm(selected_lines.items(), total=len(selected_lines)):
            obj = line[0]
            # Check if 'contentbody' key exists
            if "contentbody" in obj:
                #print("Why HOWDY DO!!!")
                # Retrieve information about the citing case
                citing_case_id = obj["itemid"]
                citing_name = obj["docname"]
                full_text = obj["contentbody"]
                
                passages = document_segmentation(citing_case_id, full_text, experiment_type, court_level)
                
                # Iterate through each passage
                for passage_type, passage in passages.items():
                    # Format the prompt with instance-specific information
                    prompt = get_prompt(citing_name, passage)
                    
                    # Create the full entry
                    batch_entry = {
                        "custom_id": f'{citing_case_id}_{passage_type}',
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": AI_model,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "response_format": {"type": "json_object"},
                            "max_tokens": token_dict[experiment_type],
                            "temperature": 0,
                            "top_p": 1,
                            "seed": 42
                        }
                    }
                    
                    # Write the batch entry directly to the output file
                    json.dump(batch_entry, outfile)
                    outfile.write("\n")
    #print(obj)

    print(f"Batch file created: {output_file}")


if __name__ == "__main__":
    # Connect to api key
    client = OpenAI(api_key=openai_key)

    # Input file - valid court levels: ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE", "4_DECGRANDCHAMBER", "5_ADMISSIBILITY", "6_ADMISSIBILITYCOM"]
    court_level = "1_GRANDCHAMBER"
    filename = f"{court_level}_meta.json"
    base_path = os.path.join(os.getcwd(), os.pardir, os.pardir, "Court_Data", "metadata", "full_case_metadata", filename)
    
    # Normalise the resulting path (optional but recommended)
    input_json_file = os.path.normpath(base_path)
    
    # Valid AI models: ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18"]
    AI_models = ["citation_annotation_project"]
        
    # Valid experiment types: ["law_section", "subsections", "bullet_passages"]
    experiment_types = ["law_section"]
    
    # Valid task types: ["citation_annotation_project"]
    task_type = "citation_annotation_project"
    case_limit = 100 # Take only the n most recent cases from the json dataset
    
    # Filter input JSON file to only include entries where "contentbody" is not empty
    valid_entries = []
    with open(input_json_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj and "contentbody" in obj and isinstance(obj["contentbody"], str) and obj["contentbody"].strip():
                    valid_entries.append(line)
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                print(f"Warning: Invalid JSON line skipped: {line.strip()}")
                continue
    
    # If the court_level requires sampling by year
    if court_level in ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE", "4_DECGRANDCHAMBER", "5_ADMISSIBILITY", "6_ADMISSIBILITYCOM"]:
        selected_lines = systematic_sample(valid_entries, case_limit)   
    else:
        raise ValueError(f"Unexpected court level: {court_level}")
    
    for AI_model in AI_models:
        for experiment_type in experiment_types:
            main(selected_lines, AI_model, court_level, experiment_type, task_type, case_limit)
