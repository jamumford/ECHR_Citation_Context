"""
Version History:
v1_2 = Extended to provide summary analysis across Article results.
v1_1 = Extended to report on individual outcome labels ["violation", "nonviolation", "unresolved"].
v1_0 = Produces summary analysis across experiment_types for case outcome results.
"""

import os
import json
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(confusion_matrix):
    # Extract the labels
    labels = list(confusion_matrix.keys())

    # Create true and predicted lists for micro-average calculations
    true_labels = []
    pred_labels = []

    for true_label, preds in confusion_matrix.items():
        for pred_label, count in preds.items():
            true_labels.extend([true_label] * count)
            pred_labels.extend([pred_label] * count)

    # Calculate macro-average metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', labels=labels
    )

    # Calculate metrics per label
    label_precision, label_recall, label_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, labels=labels
    )

    # Prepare results
    metrics = {
        "MacroAverage": {
            "Precision": macro_precision,
            "Recall": macro_recall,
            "F1": macro_f1,
        },
        "PerLabel": {},
    }

    # Add per-label metrics
    for idx, label in enumerate(labels):
        metrics["PerLabel"][label] = {
            "Precision": label_precision[idx],
            "Recall": label_recall[idx],
            "F1": label_f1[idx],
        }

    return metrics


def combine_matrices(all_confusion_matrices):

    combined_confusion_matrix = {}
    for cm in all_confusion_matrices:
        for true_label, preds in cm.items():
            if true_label == "unresolved":
                if analysis_type == "Overall":
                    true_label = "nonviolation"
                elif analysis_type == "Article":
                    true_label = "nonviolations"
                else:
                    raise ValueError(f"Unexpected analysis_type: (analysis_type)")
            if true_label not in combined_confusion_matrix:
                combined_confusion_matrix[true_label] = {}
            for pred_label, count in preds.items():
                if pred_label == "unresolved":
                    if analysis_type == "Overall":
                        pred_label = "nonviolation"
                    elif analysis_type == "Article":
                        pred_label = "nonviolations"
                    else:
                        raise ValueError(f"Unexpected analysis_type: (analysis_type)")
                if pred_label not in combined_confusion_matrix[true_label]:
                    combined_confusion_matrix[true_label][pred_label] = 0
                combined_confusion_matrix[true_label][pred_label] += count
    
    return combined_confusion_matrix


def outcomes_passages(directory, court_level, AI_model, all_confusion_matrices):
    
    # Load all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json') and filename.startswith(court_level) and AI_model in filename:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                if analysis_type == "Overall":
                    if "ConfusionMatrix" in data:
                        all_confusion_matrices.append(data["ConfusionMatrix"])
                    else:
                        raise ValueError(f"Expected ConfusionMatrix")
                elif analysis_type == "Article":
                    if "ArticleCM" in data:
                        all_confusion_matrices.append(data["ArticleCM"])
                    else:
                        raise ValueError(f"Expected ArticleCM")
                else:
                    raise ValueError(f"Unexpected analysis_type: {analysis_type}")

    # Combine all confusion matrices
    combined_confusion_matrix = combine_matrices(all_confusion_matrices)
    #print(combined_confusion_matrix)

    # Calculate metrics
    return combined_confusion_matrix


def print_summaries(metrics):

    print("Macro-Averaged Metrics:")
    for metric, value in metrics["MacroAverage"].items():
        print(f"{metric}: {value:.6f}")

    print("\nPer-Label Metrics:")
    for label, label_metrics in metrics["PerLabel"].items():
        print(f"Label: {label}")
        for metric, value in label_metrics.items():
            print(f"  {metric}: {value:.6f}")
    print("_ _ "*10)


def main(analysis_focus):

    if evaluation_type == "benchmark":
        print("\nProcessing benchmark\n")
        if analysis_focus == "court level":
            for court_level in court_levels:
                print(f"\nProcessing court_level: {court_level}\n")
                combined_confusion_matrix = outcomes_passages(analysis_path, court_level, "", [])     
                metrics = calculate_metrics(combined_confusion_matrix)
            
                # Print results
                print_summaries(metrics)
        else:
            combined_confusion_matrix = outcomes_passages(analysis_path, "", "", [])
            metrics = calculate_metrics(combined_confusion_matrix)
        
            # Print macro-averaged metrics
            print_summaries(metrics)
    
    elif analysis_focus == "passage":
        print(f"\nProcessing evaluation_type: {evaluation_type}\n")
        for experiment_type in experiment_types:
            experiment_path = os.path.join(analysis_path, experiment_type)
            print(f"\nProcessing experiment_type: {experiment_type}\n")
            all_confusion_matrices = []
            for AI_model in AI_models:
                combined_confusion_matrix = outcomes_passages(experiment_path, "", AI_model, all_confusion_matrices)
                metrics = calculate_metrics(combined_confusion_matrix)

            # Print macro-averaged metrics
            print_summaries(metrics)
    
    elif analysis_focus == "court level":
        print(f"\nProcessing evaluation_type: {evaluation_type}\n")
        for court_level in court_levels:
            print(f"\nProcessing court_level: {court_level}\n")
            all_confusion_matrices = []
            for experiment_type in experiment_types:
                for AI_model in AI_models:
                    experiment_path = os.path.join(analysis_path, experiment_type)
                    combined_confusion_matrix = outcomes_passages(experiment_path, court_level, AI_model, all_confusion_matrices)
                
            metrics = calculate_metrics(combined_confusion_matrix)

            # Print results
            print_summaries(metrics)
    
    elif analysis_focus == "nlp model":
        print(f"\nProcessing evaluation_type: {evaluation_type}\n")
        for AI_model in AI_models: 
            print(f"\nProcessing nlp model: {AI_model}\n")
            all_confusion_matrices = []
            for experiment_type in experiment_types:
                experiment_path = os.path.join(analysis_path, experiment_type)
                combined_confusion_matrix = outcomes_passages(experiment_path, "", AI_model, all_confusion_matrices)
                
            metrics = calculate_metrics(combined_confusion_matrix)

            # Print results
            print_summaries(metrics) 
    
    else:
        raise ValueError(f"Unexpected analysis_focus: {analysis_focus}")


if __name__ == "__main__":
    
    # Valid analysis types ["Overall", "Article"]
    analysis_type = "Overall"
    # Valid evaluation types ["benchmark", "no_context", "positive_context", "all_context"]
    evaluation_type = "positive_context"   
    # Valid experiment types: ["passage", "court level", "nlp model"]
    analysis_focus = "nlp model"
    
    # Specify the directory containing the JSON files
    analysis_path = os.path.join("Analysis", evaluation_type)
    # Valid experiment types: ["1_law_section", "2_subsections", "3_bullet_passages"]
    experiment_types = ["3_bullet_passages"]
    # Valid Court Levels: ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE"]
    court_levels = ["1_GRANDCHAMBER", "2_CHAMBER", "3_COMMITTEE"]    
    # Valid AI models: ['gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18']
    AI_models = ['gpt-4o-mini-2024-07-18']
    
    main(analysis_focus)
