"""
Version history
v2_0 = Adapts to "Goldilocks" experiments.
v1_1 = Accomodates article_only and article_and_context task type batches.
v1_0 = adapted from similar script s3_submit_batch_files_v2_1.py designed for the judgment and keyword
    prediction experiments for Jurix 2024, but separated to solely the download batch function (with
    submitting the batch outputs from the API being contained in a separate script 
    s2_submit_batch_files).
"""

import openai
from openai import OpenAI
from config import openai_key
import os
import time

client = OpenAI(api_key=openai_key)
  
            
def download_batch_files(court_level, experiment_name, filepath):
    '''
    Downloads files from OpenAI API after batch processing, 
    appending the batch ID to the file name for uniqueness.
    '''
    print(f"Downloading batch files for experiment: {experiment_name}")
    batches = client.batches.list()
    
    for batch in batches:
        metadata = batch.metadata.get('description', '').split(':')

        # Ensure that the metadata matches the experiment name
        if len(metadata) > 0 and metadata[0].strip() == experiment_name:
            file_id = batch.output_file_id
            if not file_id:
                print(f"No output file ID found for batch: {batch.id} with description {batch.metadata['description']}")
                continue

            try:
                # Download the batch output file content
                batch_output_file = client.files.content(file_id).content
                
                # Use metadata[1] as the base filename and append the batch ID for uniqueness
                base_filename = metadata[1].strip()
                unique_filename = f"{court_level}_{base_filename}.jsonl"  # Adjust extension if needed
                
                # Write the file with the unique filename
                with open(os.path.join(filepath, unique_filename), "wb") as f:
                    f.write(batch_output_file)
                print(f"Downloaded batch {batch.id} to {unique_filename}")
            except Exception as e:
                print(f"Error downloading file for batch {batch.id}: {e}")
 
 
def main(court_level, experiment_type, task_type):

    
    save_dir = os.path.join('Results', task_type, experiment_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    download_batch_files(court_level, experiment_name=f'{task_type}_{experiment_type}_{court_level}', filepath=save_dir)
    
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
                try:
                    main(court_level, experiment_type, task_type)
                except openai.RateLimitError as e:
                    print(f"Rate limit exceeded: {e}")
                    time.sleep(20)  # Wait for 20 seconds before retrying    
                # Sleep for a small interval between each API call to avoid hitting the rate limit
                time.sleep(6)

    