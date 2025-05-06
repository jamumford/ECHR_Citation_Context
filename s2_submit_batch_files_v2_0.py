"""
Version history
v2_0 = Submits larger batch files for the 'Goldilocks' experiments.
v1_2 = Accomodates article_only and article_and_context task type batches.
v1_1 = Corrected prompt_path to only process jsonl file corresponding to AI_model.
v1_0 = adapted from similar script s3_submit_batch_files_v2_1.py designed for the judgment and keyword
    prediction experiments for Jurix 2024, but separated to solely the submit batch function (with
    downloading the batch outputs from the API being contained in a separate script 
    s3_download_batch_files).
"""

import openai
from openai import OpenAI
from config import openai_key
import os
import time


client = OpenAI(api_key=openai_key)


def send_to_api(prompt_path, AI_model, experiment_desciption):

    '''
    Sends files to OpenAI API for batch processing

    Experiment_description naming conventions =
        'Experiment {number} {type}:'
    '''
    i = f'{AI_model}.jsonl'
    filepath = os.path.join(prompt_path, i)
    print(f'filepath: {filepath}')

    batch_input_file = client.files.create(
        file=open(filepath, "rb"),
        purpose="batch"
        )

    batch_input_file_id = batch_input_file.id

    client.batches.create(
    input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": f"{experiment_desciption} {AI_model}"
        })
 
 
def main(AI_model, court_level, experiment_type, task_type):
    
    prompt_path = os.path.join('Batches', task_type, experiment_type, court_level)
    if not os.path.exists(prompt_path):
        print(f"Valid batch file not found at {prompt_path}. Exiting function.")
        return  # Exit the function early if the file does not exist
    send_to_api(prompt_path, AI_model, experiment_desciption=f'{task_type}_{experiment_type}_{court_level}: ')
    
    return
    
    
if __name__ == '__main__':
  
    # Valid AI models: ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
    AI_models = ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
    
    # Valid Court Levels: ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE"]
    court_levels = ["2_CHAMBER", "3_COMMITTEE"]
        
    # Valid experiment types: ["law_section", "subsections", "bullet_passages"]
    experiment_types = ["law_section", "subsections", "bullet_passages"]
    
    # Valid task types: ['article_only', 'article_and_context']
    task_types = ['article_and_context']
    
    for task_type in task_types:
        for experiment_type in experiment_types:
            for court_level in court_levels:
                for AI_model in AI_models:
                    try:
                        main(AI_model, court_level, experiment_type, task_type)
                    except openai.RateLimitError as e:
                        print(f"Rate limit exceeded: {e}")
                        time.sleep(20)  # Wait for 20 seconds before retrying    
                    # Sleep for a small interval between each API call to avoid hitting the rate limit
                    time.sleep(6)

    