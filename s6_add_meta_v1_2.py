"""
Version History:
v1_2 = Adjusted to accommodate different court_level experiment outputs. Also added kpdate field.
v1_1 = Stores metadata as dictionary rather than separate variables to reduce clutter.
v1_0 = Checks for duplicate citing case ids and then adds required citing case metadata for 
    classification task.
"""

from collections import Counter
import json
import os


def find_match(citing_id, metadata):            

    # Check if "docname" matches case_name
    for obj in metadata:
        metadict = {}
        if obj.get("itemid") == citing_id:
            metadict["violations"] = obj.get("violation", [])
            metadict["nonviolations"] = obj.get("nonviolation", [])
            metadict["articles raised"] = obj.get("article", [])
            metadict["importance"] = obj["importance"]
            metadict["kpdate"] = obj["kpdate"]
            
            return metadict
    #print(cited_case_names)                    
    raise ValueError(f"No matching case found for citing case: {citing_id}")


def main(jsonl_file, metadata):
    print(f"\nProcessing file for duplicate case IDs: {jsonl_file}")
    
    # Automatically generate output file name by replacing .jsonl with .json
    if not jsonl_file.endswith('.jsonl'):
        raise ValueError("Input file must have a .jsonl extension")
    output_file = jsonl_file.replace('_updated.jsonl', '.json')
    
    # Counter to track occurrences of case IDs
    case_id_counter = Counter()
    
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        # Read the entire file
        content = file.read()
        
        # Parse the JSON content
        data = json.loads(content)

        # Iterate through the top-level keys
        for case_id in data.keys():
            case_id_counter[case_id] += 1
            
            # Fetch violations, nonviolations and all articles raised for the case_id
            metadict = find_match(case_id, metadata)

            # Update the data dictionary with violations and nonviolations
            for key in metadict:
                data[case_id][key] = metadict[key]
    
    # Check for duplicates
    duplicates = [case_id for case_id, count in case_id_counter.items() if count > 1]
    
    if duplicates:
        print(f"Duplicate case IDs found: {duplicates}")
    else:
        print("No duplicate case IDs found.")
    
    # Save the updated data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file, ensure_ascii=False, indent=4)
    
    print(f"Updated file saved to: {output_file}")


if __name__ == "__main__":

    # Identify and normalise the metadata path
    base_meta_path = os.path.join(os.getcwd(), os.pardir, os.pardir, "Court_Data", "metadata", "full_case_metadata")
    meta_dir = os.path.normpath(base_meta_path)
    
    # Results file
    results_dir = os.path.join('Results', 'article_and_context')
    
    # Valid AI models: ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
    AI_models = ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
    
    # Valid Court Levels: ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE"]
    court_levels = ["2_CHAMBER", "3_COMMITTEE"]
    
    # Valid experiment types: ["law_section", "subsections", "bullet_passages"]
    experiment_types = ["law_section", "subsections", "bullet_passages"]
    
    for court_level in court_levels:
        # Pre-load the metadata JSON file
        meta_file = os.path.join(meta_dir, f"{court_level}_meta.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = [json.loads(line) for line in f]
        for experiment_type in experiment_types:        
            for AI_model in AI_models:
                jsonl_file = os.path.join(results_dir, experiment_type, f"{court_level}_{AI_model}_updated.jsonl")
                main(jsonl_file, metadata)
