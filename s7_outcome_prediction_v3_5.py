"""
Version History:
v3_5 = Saves Article results confusion matrix.
v3_4 = Added benchmark classifier based on extractedappnos.
v3_3 = Extended to include categorisation by distinguishing labels.
v3_2 = Adjusted to produce results for different court levels. Also prints output to file
v3_1 = Expanded analysis to cover overall case outcome classifications.
v3_0 = Adapted for 'Goldilocks' experiments.
v2_0 = Performs analysis on NLP results.
v1_3 = Added overall binary classification for each result item and related summary statistics.
v1_2 = Produces summary performance stats over dataset.
v1_1 = Extended to accommodate nearest Article metadata.
v1_0 = Counts and compares Article outcomes in a case with the outcomes for the cited cases.
"""

from collections import defaultdict
import io
import json
import os
from sklearn.metrics import classification_report


def amend_counts(citation_info, articles_raised):
    prediction_classes = {}
    for class_type in class_types:
        prediction_classes[class_type] = []
    for article in articles_raised:
        vio_count = citation_info[article]["violations"]
        non_count = citation_info[article]["nonviolations"]
        unres_count = citation_info[article]["unresolved"]
        
        if vio_count == 0 and non_count == 0:
            prediction_classes["unresolved"].append(article)
        elif vio_count >= non_count:
            prediction_classes["violations"].append(article)
        elif vio_count < non_count:
            prediction_classes["nonviolations"].append(article)
        else:
            raise ValueError('Unexpected count outcome')
    
    return prediction_classes


def extracted_appno_output(data):

    # Initialise results data
    results = {}

    # Process each object in the NLP output
    for citing_id, citing_info in data.items():
        
        results, articles_raised = initialise_results(results, citing_id, citing_info)
                
        extracted_apps = itemid_to_extracted[citing_id]
        
        for cited_app in extracted_apps:
            try:
                citation = appno_to_outcomes[cited_app]
            except:
                continue        
            articles = prune_articles(citation["articles raised"])
            cited_violations = prune_articles(citation["violation"])
            cited_nonviolations = prune_articles(citation["nonviolation"])
            
            # Find the intersection of Articles raised by citing case and those found by NLP in the citation instance
            intersection = list(set(articles) & set(articles_raised))
            
            # Iterate over articles
            for article in intersection:
                class_type = evaluate_labels("no_context", "positive", article, cited_violations, cited_nonviolations)

                # Update nlp_counts if appropriate
                if class_type != "no_influence":
                    results[citing_id][article][class_type] += 1  
        
        # Get nlp predictions
        results[citing_id]["predictions"] = amend_counts(results[citing_id], articles_raised)
        results[citing_id]["predictions"] = overall_outcome(results[citing_id]["predictions"])
        
    return results


def evaluate_labels(evaluation_type, context_class, article, cited_violations, cited_nonviolations):

    # If evaluation_type == "no_context", then the context is effectively treated as "positive"
    if evaluation_type == "no_context":
        context_class = "positive"

    if context_class == "positive":
        if article in cited_violations:
            class_type = "violations"
        elif article in cited_nonviolations:
            class_type = "nonviolations"
        else:
            class_type = "unresolved"
    
    elif context_class == "negative":
        if evaluation_type == "all_context":
            if article in cited_violations:
                class_type = "nonviolations"
            elif article in cited_nonviolations:
                class_type = "violations"
            else:
                class_type = "no_influence"
        else:
            class_type = "no_influence"
    
    elif context_class == "neutral":
        class_type = "no_influence"
    
    else:
        raise ValueError(f"Unexpected context_class: {context_class}")
    
    return class_type  


def initialise_results(results, citing_id, citing_info):   

    results[citing_id] = {}
    
    # Get citing case 'True' classifications and other info
    articles_raised = prune_articles(citing_info["articles raised"])
    results[citing_id]["violations"] = prune_articles(citing_info["violations"])
    results[citing_id]["nonviolations"] = prune_articles(citing_info["nonviolations"])
    results[citing_id]["unresolved"] = []
    for article in articles_raised:
        if article not in results[citing_id]["violations"] + results[citing_id]["nonviolations"]:
            results[citing_id]["unresolved"].append(article)
    
    # Update results[citing_id] with overall outcome classification
    results[citing_id] = overall_outcome(results[citing_id])
    
    # Initialise classification outputs                      
    for article in articles_raised:
        results[citing_id][article] = {}
        for class_type in class_types:
            results[citing_id][article][class_type] = 0  
    
    return results, articles_raised
    

def nlp_returns(data):

    # Initialise results data
    results = {}

    # Process each object in the NLP output
    for citing_id, citing_info in data.items():
    
        # Get nlp output list
        nlp_output = []
        # Get nlp output classes
        for idx, batch_output in enumerate(citing_info["nlp outputs"]):  
            try:              
                nlp_output += batch_output["response"]["body"]["choices"][0]["message"]["content"]["citations"]
            except (TypeError, KeyError, IndexError) as e:
                # Skip to the next item if the expected structure is not found
                #print(f"Skipping batch {citing_info['nlp outputs'][idx]['custom_id']} due to error: {e}")
                continue
        
        if len(nlp_output) == 0:
            print(f"Skipping citing case {citing_id} due to no valid nlp batch outputs")
            continue
            
        results, articles_raised = initialise_results(results, citing_id, citing_info)
        
        #print(nlp_output)
        # Extract citation details
        for citation in nlp_output:
            articles = prune_articles(citation["articles"])
            try:
                context_class = citation["context_class"].lower()
            except:
                print(f"no context_class for item in citing_info {citation}")
            if context_class not in ['positive', 'negative', 'neutral']:
                raise ValueError(f'Unexpected context class for citation: {context_class}')
            cited_violations = prune_articles(citation["violation"])
            cited_nonviolations = prune_articles(citation["nonviolation"])
            
            # Find the intersection of Articles raised by citing case and those found by NLP in the citation instance
            intersection = list(set(articles) & set(articles_raised))
            
            # Iterate over articles
            for article in intersection:
                class_type = evaluate_labels(evaluation_type, context_class, article, cited_violations, cited_nonviolations)

                # Update nlp_counts if appropriate
                if class_type != "no_influence":
                    results[citing_id][article][class_type] += 1  
        
        # Get nlp predictions
        results[citing_id]["predictions"] = amend_counts(results[citing_id], articles_raised)
        results[citing_id]["predictions"] = overall_outcome(results[citing_id]["predictions"])
        
    return results


def overall_outcome(classification_counts):

    if len(classification_counts["violations"]) > 0:
        classification_counts["outcome"] = "violation"
    elif len(classification_counts["nonviolations"]) > 0:
        classification_counts["outcome"] = "nonviolation"
    else:
        classification_counts["outcome"] = "unresolved"
    
    return classification_counts


def preload_metadata(meta_dir):
    """
    Preloads all metadata from JSON files in the directory and returns a dictionary indexed by key_variable.
    """
    itemid_to_extracted = {}
    appno_to_outcomes = {}
    for filename in os.listdir(meta_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(meta_dir, filename)
            print(f"Examining file_path: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                metadata = [json.loads(line) for line in f]
                for obj in metadata:
                    itemid_key = obj.get("itemid", "")
                    appno_key = obj.get("appno", "")
                    
                    # Update itemid_to_extracted dictionary
                    if itemid_key not in itemid_to_extracted:  # Only add if not already present
                        extracted_appnos = obj.get("extractedappno", "")
                        if extracted_appnos:
                            itemid_to_extracted[itemid_key] = extracted_appnos.split(";")
                    
                    # Update appno_to_outcomes dictionary
                    split_appnos = appno_key.split(";")
                    for appkey in split_appnos:
                        if appkey not in appno_to_outcomes:  # Only add if not already present
                            appno_to_outcomes[appkey] = {
                                "violation": obj.get("violation", []),
                                "nonviolation": obj.get("nonviolation", []),
                                "articles raised": obj.get("article", []),
                            }
                        
    return itemid_to_extracted, appno_to_outcomes
    

def prune_articles(article_list):
    """
    Prunes an input article string, retaining only articles of the form 'a' or 'Pb-c'.
    
    Args:
        article_string (str): A string of articles delimited by semi-colons (e.g., "8;8-1;6;6-1").
    
    Returns:
        list: A list of pruned articles matching the required formats.
    """
    pruned_articles = []
    if isinstance(article_list, str):
        articles = article_list.split(";")  # Split the string into individual articles
    elif isinstance(article_list, list):
        articles = article_list
    else:
        raise TypeError(f"Unexpected article list type: {type(article_lists)}")
    for article in articles:
        if article.isdigit() or (
            article.startswith("P") and
            len(parts := article[1:].split("-")) == 2 and
            all(part.isdigit() for part in parts)
        ):
            pruned_articles.append(article)
    return pruned_articles


def stats_printing(experiment_name, experiment_type, nlp_article_stats, nlp_overall_stats, outcome_cm, article_cm):
    """
    Print summary statistics and save the output in a JSON format for easy loading.

    Args:
        experiment_name (str): Name of the experiment (e.g., experiment type and model).
        nlp_article_stats (dict): Per-article statistics.
        nlp_overall_stats (dict): Overall statistics.
        outcome_cm (dict): Confusion matrix comparing true and predicted overall outcomes.
        article_cm (dict): Confusion matrix comparing true and predicted specific article outcomes.
    """
    # Analysis save path
    analysis_dir = os.path.join("Analysis", "article_and_context", experiment_type)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # File to save the results
    output_file = os.path.join(analysis_dir, f"{court_level}_{AI_model}.json")

    # Prepare output dictionary
    output_data = {
        "ExperimentName": experiment_name,
        "PerArticleStatistics": {},
        "OverallStatistics": {},
        "OutcomeClassificationStatistics": {},
        "OverallOutcomeStatistics": {},
        "ConfusionMatrix": outcome_cm,
        "ArticleCM": article_cm,
    }

    # Process per-article statistics
    for label, stats in nlp_article_stats.items():
        precision = stats["TP"] / (stats["TP"] + stats["FP"]) if (stats["TP"] + stats["FP"]) > 0 else 0.0
        recall = stats["TP"] / (stats["TP"] + stats["FN"]) if (stats["TP"] + stats["FN"]) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        output_data["PerArticleStatistics"][label] = {
            "TP": stats["TP"],
            "FP": stats["FP"],
            "FN": stats["FN"],
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }

    # Process overall statistics
    precision = nlp_overall_stats["TP"] / (nlp_overall_stats["TP"] + nlp_overall_stats["FP"]) if (nlp_overall_stats["TP"] + nlp_overall_stats["FP"]) > 0 else 0.0
    recall = nlp_overall_stats["TP"] / (nlp_overall_stats["TP"] + nlp_overall_stats["FN"]) if (nlp_overall_stats["TP"] + nlp_overall_stats["FN"]) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    output_data["OverallStatistics"] = {
        "TP": nlp_overall_stats["TP"],
        "FP": nlp_overall_stats["FP"],
        "FN": nlp_overall_stats["FN"],
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    # Process outcome classification statistics
    labels = ["violation", "nonviolation", "unresolved"]
    outcome_stats = {}

    for label in labels:
        TP = outcome_cm[label][label]
        FP = sum(outcome_cm[pred_label][label] for pred_label in labels if pred_label != label)
        FN = sum(outcome_cm[label][pred_label] for pred_label in labels if pred_label != label)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        outcome_stats[label] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }

    output_data["OutcomeClassificationStatistics"] = outcome_stats

    # Calculate overall outcome statistics across all labels
    # Calculate macro precision, recall, and F1-score
    macro_precision = sum(stats["Precision"] for stats in outcome_stats.values()) / len(outcome_stats)
    macro_recall = sum(stats["Recall"] for stats in outcome_stats.values()) / len(outcome_stats)
    macro_f1 = sum(stats["F1"] for stats in outcome_stats.values()) / len(outcome_stats)
    
    # Add macro metrics to the output data
    output_data["OverallOutcomeStatistics"] = {
        "Precision": macro_precision,
        "Recall": macro_recall,
        "F1": macro_f1,
    }

    # Save to JSON file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    # Optionally, print a message confirming the save
    print(f"Statistics saved to {output_file}")


def summary_statistics(results):
    """
    Generate summary statistics for predictions compared to true classes.

    Args:
        results (list): A list of result dictionaries containing true and predicted classes.

    Returns:
        tuple: A dictionary with per-article statistics, a dictionary with overall statistics, 
               a confusion matrix comparing true and predicted outcomes, and a confusion matrix for article-level stats.
    """
    article_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    overall_article_stats = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    # Initialise the confusion matrices
    outcome_confusion_matrix = defaultdict(lambda: defaultdict(int))
    article_confusion_matrix = defaultdict(lambda: defaultdict(int))

    for citing_id, result in results.items():
        true_violations = set(result["violations"])
        true_nonviolations = set(result["nonviolations"])
        true_unresolved = set(result["unresolved"])
        true_outcome = result["outcome"]

        pred_violations = set(result["predictions"]["violations"])
        pred_nonviolations = set(result["predictions"]["nonviolations"])
        pred_unresolved = set(result["predictions"]["unresolved"])
        pred_outcome = result["predictions"]["outcome"]

        # For each true label (violations, nonviolations, unresolved), compare to predicted labels
        for true_label, true_set in zip(
            ["violations", "nonviolations", "unresolved"],
            [true_violations, true_nonviolations, true_unresolved]
        ):
            for pred_label, pred_set in zip(
                ["violations", "nonviolations", "unresolved"],
                [pred_violations, pred_nonviolations, pred_unresolved]
            ):
                intersection_count = len(true_set & pred_set)
                article_confusion_matrix[true_label][pred_label] += intersection_count

        # For each label (violations, nonviolations, unresolved), calculate stats
        for label, true_set, pred_set in zip(
            ["violations", "nonviolations", "unresolved"],
            [true_violations, true_nonviolations, true_unresolved],
            [pred_violations, pred_nonviolations, pred_unresolved]
        ):
            TP = len(true_set & pred_set)
            FP = len(pred_set - true_set)
            FN = len(true_set - pred_set)
            TN = len(results) - (TP + FP + FN)  # Total cases minus incorrect/predicted counts

            article_stats[label]["TP"] += TP
            article_stats[label]["FP"] += FP
            article_stats[label]["FN"] += FN
            article_stats[label]["TN"] += TN

            overall_article_stats["TP"] += TP
            overall_article_stats["FP"] += FP
            overall_article_stats["FN"] += FN
            overall_article_stats["TN"] += TN

        # Update the confusion matrix for outcomes
        outcome_confusion_matrix[true_outcome][pred_outcome] += 1

    return article_stats, overall_article_stats, outcome_confusion_matrix, article_confusion_matrix
  

def main(json_data):
        
    with open(json_data, 'r', encoding='utf-8') as file:
        # Read the entire file
        content = file.read()
        
        # Parse the JSON content
        data = json.loads(content)
    
    print(f"Processing json file: {json_data}")
    
    # Update citation counts
    if evaluation_type == "benchmark":
        results = extracted_appno_output(data)
        json_data = os.path.join(results_dir, experiment_type, f"{court_level}.json")
    else:
        results = nlp_returns(data)   
    
    # Generate summary statistics for predictions
    nlp_article_stats, nlp_overall_stats, outcome_cm, article_cm = summary_statistics(results)
    stats_printing(json_data, experiment_type, nlp_article_stats, nlp_overall_stats, outcome_cm, article_cm)


if __name__ == "__main__":

    # Identify and normalise the metadata path
    base_meta_path = os.path.join(os.getcwd(), os.pardir, os.pardir, "Court_Data", "metadata", "full_case_metadata")
    meta_dir = os.path.normpath(base_meta_path)
    
    # Results directory
    results_dir = os.path.join('Results', 'article_and_context')
    
    # Valid AI models: ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
    AI_models = ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
        
    # Valid experiment types: ["law_section", "subsections", "bullet_passages"]
    experiment_types = ["law_section", "subsections", "bullet_passages"]
    
    # Valid Court Levels: ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE"]
    court_levels = ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE"]
    
    # Initialise output classes
    class_types = ["violations", "nonviolations", "unresolved"]
    
    # Valid evaluation types ["benchmark", "no_context", "positive_context", "all_context"]
    evaluation_type = "all_context"
    
    # Produce classifications
    if evaluation_type == "benchmark":
        itemid_to_extracted, appno_to_outcomes = preload_metadata(meta_dir)
        for court_level in court_levels:
            # Select law_section for easy case_id extraction
            experiment_type = evaluation_type
            AI_model = ""
            json_data = os.path.join(results_dir, "law_section", f"{court_level}_gpt-4o-mini-2024-07-18.json")
            main(json_data)
    else:
        for experiment_type in experiment_types:
            print(f"\nProcessing results for {experiment_type}")    
            for AI_model in AI_models:  
                for court_level in court_levels:                
                    json_data = os.path.join(results_dir, experiment_type, f"{court_level}_{AI_model}.json")
                    main(json_data)
