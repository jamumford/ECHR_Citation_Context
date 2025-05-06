"""
Version History:
v2_3 = Adjusted to accommodate different court_level experiment outputs.
v2_2 = Overhauled code to make it more efficient.
v2_1 = Added citing case info to output.
v2_0 = Converted to process article and context extraction, finding metadata for
    retrieved cited cases.
v1_0 = Formats OpenAI output for easier evaluation.
"""

from collections import defaultdict
import json
import os
from tqdm import tqdm


class CaseInsensitiveDict(defaultdict):
    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __contains__(self, key):
        return super().__contains__(key.lower())


def extract_outcomes(nlp_output, metadata_dict, custom_id, cited_case_names):
    """
    Extracts and appends the article outcomes for cited cases to the nlp_output.
    """
    try:
        citations_str = nlp_output[0]["message"]["content"]
        citations_data = json.loads(citations_str)
    except json.JSONDecodeError as e:
        print(f"Malformed JSON for {custom_id}: {e.msg}")
        return None, cited_case_names

    if "citations" not in citations_data:
        raise ValueError("No citations found in the provided data.")

    extracted_citations = citations_data["citations"]

    for citation in extracted_citations:
        cited_case_name = citation.get("case_name", "N/A")
        violation, nonviolation, matched_name = find_match(cited_case_name, cited_case_names, metadata_dict)
        citation.update({
            "violation": violation,
            "nonviolation": nonviolation,
        })
        cited_case_names[matched_name] = {"violation": violation, "nonviolation": nonviolation}

    citations_data["citations"] = extracted_citations
    nlp_output[0]["message"]["content"] = citations_data
    
    return nlp_output, cited_case_names


def find_match(cited_case_name, cited_case_names, metadata_dict):
    violation = False
    nonviolation = False

    # Check for a match in cited_case_names (case-insensitive handled automatically)
    if cited_case_name in cited_case_names:
        case_data = cited_case_names[cited_case_name]
        violation = case_data.get("violation", [])
        nonviolation = case_data.get("nonviolation", [])
        return violation, nonviolation, cited_case_name

    # Check the preloaded metadata for a match
    case_key = f"case of {cited_case_name}".lower()  # Keep metadata key handling as-is
    if case_key in metadata_dict:
        case_data = metadata_dict[case_key]
        violation = case_data["violation"]
        nonviolation = case_data["nonviolation"]
        return violation, nonviolation, cited_case_name

    # If no match is found
    return "", "", ""

    

def preload_metadata(meta_dir):
    """
    Preloads all metadata from JSON files in the directory and returns a dictionary indexed by case name.
    """
    metadata_dict = {}
    for filename in os.listdir(meta_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(meta_dir, filename)
            print(f"Examining file_path: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                metadata = [json.loads(line) for line in f]
                for obj in metadata:
                    docname = obj.get("docname", "").lower()
                    if docname not in metadata_dict:  # Only add if not already present
                        metadata_dict[docname] = {
                            "violation": obj.get("violation", []),
                            "nonviolation": obj.get("nonviolation", []),
                        }
                        
    return metadata_dict
                

def main(experiment_type, court_level, AI_model, filedir):
            
    # Load the results JSONL file
    file_name = f"{court_level}_{AI_model}.jsonl"
    result_file_path = os.path.join(filedir, file_name)
    print(f"Processing results file: {result_file_path}")
    with open(result_file_path, "r", encoding="utf-8") as infile:
        data = [json.loads(line) for line in infile]
        
    # Initialise citing_case_id to later check for repeated and abbreviated citations
    citing_case_id = None
    
    # Initialise dictionary for results data organised by citing cases
    formatted_data = {}
    
    # Iterate through outputs and update with extracted passages
    for item in tqdm(data):
    
        custom_id = item["custom_id"]
        custom_id_parts = custom_id.split("_")
        if len(custom_id_parts) != 2:
            raise ValueError(f"Invalid custom_id format: {custom_id}")
        current_id = custom_id_parts[0]
        
        # Refresh cited_case_names if processing a new citing case
        if current_id != citing_case_id:
            citing_case_id = current_id
            cited_case_names = CaseInsensitiveDict()
        
        # Ensure the citing_case_id exists as a key in the formatted_data dictionary
        if citing_case_id not in formatted_data:
            formatted_data[citing_case_id] = {"nlp outputs":[]}
    
        finish_reason = item["response"]["body"]["choices"][0]["finish_reason"]
        if finish_reason == "stop":
            amended_output, cited_case_names = extract_outcomes(item["response"]["body"]["choices"], metadata_dict, custom_id, cited_case_names)
            if amended_output:
                item["response"]["body"]["choices"] = amended_output
            else:
                continue
        elif finish_reason != "length":
            print(f"Unexpected finish reason: {finish_reason}")
            continue
    
        # Append the updated item to the list of the current citing_case_id
        formatted_data[citing_case_id]["nlp outputs"].append(item)
    
    # Save the updated data back to a new JSONL file
    output_file = os.path.join(filedir, f"{court_level}_{AI_model}_updated.jsonl")
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write(json.dumps(formatted_data, indent=4) + '\n')
        print(f'Updated data saved to: {output_file}\n')
    except IOError as e:
        print(f"Failed to save updated data: {e}\n")


if __name__ == "__main__":
    
    # Identify and normalise the metadata path
    base_meta_path = os.path.join(os.getcwd(), os.pardir, os.pardir, "Court_Data", "metadata", "full_case_metadata")
    meta_dir = os.path.normpath(base_meta_path)
    metadata_dict = preload_metadata(meta_dir)

    # Results directory
    results_dir = os.path.join('Results', 'article_and_context')
    
    # Valid AI models: ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
    AI_models = ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
    
    # Valid Court Levels: ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE"]
    court_levels = ["2_CHAMBER", "3_COMMITTEE"]
    
    # Valid experiment types: ["law_section", "subsections", "bullet_passages"]
    experiment_types = ["law_section", "subsections", "bullet_passages"]
    
    for experiment_type in experiment_types:
        for court_level in court_levels:
            for AI_model in AI_models:
                filedir = os.path.join(results_dir, experiment_type)
                #print(f"filepath: {filepath}")
                main(experiment_type, court_level, AI_model, filedir)
