"""
Evaluation Part 1: Model & Pipeline Comparison on MentalChat16K Test Set

Tests different models and pipeline configurations:
1. Base Mistral-7B-Instruct (direct response)
2. Full 3-component pipeline (Clinical -> Opinion -> Response)
3. No Clinical pipeline (Input -> Opinion -> Response)
4. No Opinion pipeline (Clinical -> Response)
5. Finetuned MentalChat-16K model (direct response)

All evaluated on the MentalChat16K test set with GPT-4 judging.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Import centralized judging
from judging import evaluate_response, METRICS

# Import pipeline components
from pipeline_pieces import (
    run_clinical_description,
    run_professional_opinion,
    run_final_response
)


def load_mentalchat_test_set(num_questions: int = None) -> List[Dict[str, str]]:
    """
    Load MentalChat16K dataset from HuggingFace.

    Args:
        num_questions: Number of questions to load (None = all)

    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    print("Loading MentalChat16K dataset from HuggingFace...")

    # Load the dataset (only has 'train' split)
    dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")

    # Use the last 20% as a held-out test set
    total_size = len(dataset)
    test_start = int(total_size * 0.8)
    test_dataset = dataset.select(range(test_start, total_size))

    print(f"Using last 20% of dataset as test set ({len(test_dataset)} examples)")

    questions = []
    for item in test_dataset:
        questions.append({
            "question": item["Context"],
            "reference_answer": item["Response"]
        })

        if num_questions and len(questions) >= num_questions:
            break

    print(f"Loaded {len(questions)} questions for evaluation")
    return questions


def generate_mistral(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate response using Mistral instruction format."""
    formatted_prompt = f"[INST] {prompt} [/INST]"

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


def generate_llama(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate response using Llama chat format."""
    system = """You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description.
The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ]

    # Format with chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

    return response


def evaluate_base_mistral(questions: List[Dict], model, tokenizer) -> List[Dict]:
    """Evaluate base Mistral-7B-Instruct (direct response)."""
    print("\n" + "="*80)
    print("EVALUATING: Base Mistral-7B-Instruct (Direct Response)")
    print("="*80)

    results = []
    for i, item in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}")
        question = item['question']

        # Direct response
        start = time.time()
        response = generate_mistral(model, tokenizer, question)
        gen_time = time.time() - start

        # Judge
        eval_result = evaluate_response(question, response, "Base Mistral-7B")

        results.append({
            "question": question,
            "response": response,
            "generation_time": gen_time,
            "scores": eval_result['scores'],
            "explanation": eval_result['explanation'],
            "average": eval_result['average']
        })

        print(f"  Generated in {gen_time:.2f}s")

    return results


def evaluate_full_pipeline(questions: List[Dict], model, tokenizer) -> List[Dict]:
    """Evaluate full 3-component pipeline."""
    print("\n" + "="*80)
    print("EVALUATING: Full 3-Component Pipeline")
    print("="*80)

    results = []
    for i, item in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}")
        question = item['question']

        start = time.time()

        # Component 1: Clinical description
        clinical = run_clinical_description(model, tokenizer, generate_mistral, question)

        # Component 2: Professional opinion
        opinion = run_professional_opinion(model, tokenizer, generate_mistral, clinical)

        # Component 3: Final response
        response = run_final_response(model, tokenizer, generate_mistral, question, opinion, "professional")

        gen_time = time.time() - start

        # Judge
        eval_result = evaluate_response(question, response, "Full Pipeline")

        results.append({
            "question": question,
            "clinical_description": clinical,
            "professional_opinion": opinion,
            "response": response,
            "generation_time": gen_time,
            "scores": eval_result['scores'],
            "explanation": eval_result['explanation'],
            "average": eval_result['average']
        })

        print(f"  Generated in {gen_time:.2f}s")

    return results


def evaluate_no_clinical(questions: List[Dict], model, tokenizer) -> List[Dict]:
    """Evaluate pipeline without clinical description."""
    print("\n" + "="*80)
    print("EVALUATING: No Clinical Pipeline (Input -> Opinion -> Response)")
    print("="*80)

    results = []
    for i, item in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}")
        question = item['question']

        start = time.time()

        # Component 1: Professional opinion (directly from input)
        opinion = run_professional_opinion(model, tokenizer, generate_mistral, question)

        # Component 2: Final response
        response = run_final_response(model, tokenizer, generate_mistral, question, opinion, "professional")

        gen_time = time.time() - start

        # Judge
        eval_result = evaluate_response(question, response, "No Clinical")

        results.append({
            "question": question,
            "professional_opinion": opinion,
            "response": response,
            "generation_time": gen_time,
            "scores": eval_result['scores'],
            "explanation": eval_result['explanation'],
            "average": eval_result['average']
        })

        print(f"  Generated in {gen_time:.2f}s")

    return results


def evaluate_no_opinion(questions: List[Dict], model, tokenizer) -> List[Dict]:
    """Evaluate pipeline without professional opinion."""
    print("\n" + "="*80)
    print("EVALUATING: No Opinion Pipeline (Clinical -> Response)")
    print("="*80)

    results = []
    for i, item in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}")
        question = item['question']

        start = time.time()

        # Component 1: Clinical description
        clinical = run_clinical_description(model, tokenizer, generate_mistral, question)

        # Component 2: Final response (using clinical as context)
        response = run_final_response(model, tokenizer, generate_mistral, question, clinical, "clinical")

        gen_time = time.time() - start

        # Judge
        eval_result = evaluate_response(question, response, "No Opinion")

        results.append({
            "question": question,
            "clinical_description": clinical,
            "response": response,
            "generation_time": gen_time,
            "scores": eval_result['scores'],
            "explanation": eval_result['explanation'],
            "average": eval_result['average']
        })

        print(f"  Generated in {gen_time:.2f}s")

    return results


def evaluate_finetuned_mentalchat(questions: List[Dict]) -> List[Dict]:
    """Evaluate finetuned MentalChat-16K model."""
    print("\n" + "="*80)
    print("EVALUATING: Finetuned MentalChat-16K (Llama-3.2-1B)")
    print("="*80)

    # Load finetuned model
    print("Loading finetuned model...")
    base_model_name = "unsloth/Llama-3.2-1B-Instruct"
    model_name = "khazarai/MentalChat-16K"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    model = PeftModel.from_pretrained(base_model, model_name)
    print("Finetuned model loaded!")

    results = []
    for i, item in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}")
        question = item['question']

        # Direct response
        start = time.time()
        response = generate_llama(model, tokenizer, question)
        gen_time = time.time() - start

        # Judge
        eval_result = evaluate_response(question, response, "MentalChat-16K")

        results.append({
            "question": question,
            "response": response,
            "generation_time": gen_time,
            "scores": eval_result['scores'],
            "explanation": eval_result['explanation'],
            "average": eval_result['average']
        })

        print(f"  Generated in {gen_time:.2f}s")

    # Clean up
    del model
    del base_model
    del tokenizer
    torch.cuda.empty_cache()

    return results


def debug_results(results: List[Dict], config_name: str):
    """Debug utility to inspect result structure."""
    print(f"\n{'='*80}")
    print(f"DEBUG: Inspecting {config_name} results structure")
    print(f"{'='*80}")

    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Keys: {list(result.keys())}")

        if 'scores' in result:
            if isinstance(result['scores'], dict):
                print(f"  Score keys: {list(result['scores'].keys())}")
                # Show first few scores
                sample_scores = dict(list(result['scores'].items())[:3])
                print(f"  Sample scores: {sample_scores}")
            else:
                print(f"  ERROR: 'scores' is not a dict, it's {type(result['scores'])}")
                print(f"  Value: {result['scores']}")
        else:
            print(f"  ERROR: No 'scores' key found!")
            print(f"  Available keys: {list(result.keys())}")

    print(f"{'='*80}\n")


def compute_average_scores(results: List[Dict]) -> tuple:
    """
    Compute average scores across all questions, handling missing/malformed data.

    Returns:
        Tuple of (avg_scores dict, overall_avg float, valid_count, invalid_count)
    """
    # Initialize metric collection
    metric_scores = {metric: [] for metric in METRICS}

    valid_count = 0
    invalid_count = 0

    # Collect scores, handling missing data
    for i, result in enumerate(results):
        # Check if result has scores
        if 'scores' not in result:
            print(f"  WARNING: Result {i+1} missing 'scores' key")
            invalid_count += 1
            continue

        scores = result['scores']

        # Check if scores is a dict
        if not isinstance(scores, dict):
            print(f"  WARNING: Result {i+1} 'scores' is not a dict: {type(scores)}")
            invalid_count += 1
            continue

        # Try to extract each metric
        has_all_metrics = True
        for metric in METRICS:
            if metric in scores:
                try:
                    score_value = float(scores[metric])
                    metric_scores[metric].append(score_value)
                except (ValueError, TypeError):
                    print(f"  WARNING: Result {i+1} metric '{metric}' has invalid value: {scores[metric]}")
                    has_all_metrics = False
                    break
            else:
                print(f"  WARNING: Result {i+1} missing metric '{metric}'")
                print(f"    Available keys: {list(scores.keys())}")
                has_all_metrics = False
                break

        if has_all_metrics:
            valid_count += 1
        else:
            invalid_count += 1

    # Compute averages
    avg_scores = {}
    for metric in METRICS:
        if metric_scores[metric]:
            avg_scores[metric] = sum(metric_scores[metric]) / len(metric_scores[metric])
        else:
            avg_scores[metric] = 0.0
            print(f"  WARNING: No valid scores for '{metric}'")

    # Overall average
    if avg_scores and any(v > 0 for v in avg_scores.values()):
        overall = sum(avg_scores.values()) / len(avg_scores)
    else:
        overall = 0.0

    # Compute average generation time
    valid_times = [r['generation_time'] for r in results if 'generation_time' in r]
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0.0
    avg_scores['avg_generation_time'] = avg_time

    return avg_scores, overall, valid_count, invalid_count


def print_summary(all_results: Dict[str, List[Dict]]):
    """Print summary comparison of all configurations."""
    print("\n" + "="*80)
    print("SUMMARY: Average Scores Across All Configurations")
    print("="*80)

    # First, run debug on all results
    for config_name, results in all_results.items():
        debug_results(results, config_name)

    # Now compute and print summaries
    for config_name, results in all_results.items():
        avg_scores, overall, valid_count, invalid_count = compute_average_scores(results)

        print(f"\n{config_name}:")
        print(f"  Valid/Invalid: {valid_count}/{invalid_count} results")
        print(f"  Overall Average: {overall:.2f}/10")
        print(f"  Generation Time: {avg_scores['avg_generation_time']:.2f}s")
        print(f"\n  Individual Metrics:")
        for metric in METRICS:
            print(f"    {metric}: {avg_scores[metric]:.2f}/10")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)

    config_names = list(all_results.keys())
    if not config_names:
        return

    # Print header
    print(f"\n{'Metric':<45}", end="")
    for name in config_names:
        print(f" {name[:15]:>15}", end="")
    print()
    print("-" * (45 + 16 * len(config_names)))

    # Compute all averages
    all_avgs = {}
    for config_name, results in all_results.items():
        avg_scores, overall, _, _ = compute_average_scores(results)
        all_avgs[config_name] = (avg_scores, overall)

    # Print each metric
    for metric in METRICS:
        print(f"{metric:<45}", end="")
        for config_name in config_names:
            avg_scores, _ = all_avgs[config_name]
            score = avg_scores[metric]
            print(f" {score:>14.2f}", end="")
        print()

    # Print overall
    print("-" * (45 + 16 * len(config_names)))
    print(f"{'OVERALL AVERAGE':<45}", end="")
    for config_name in config_names:
        _, overall = all_avgs[config_name]
        print(f" {overall:>14.2f}", end="")
    print()

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models and pipeline configurations on MentalChat16K test set'
    )
    parser.add_argument(
        '--num-questions',
        type=int,
        default=10,
        help='Number of questions to evaluate (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_part1_results.json',
        help='Output file for results (default: evaluation_part1_results.json)'
    )

    args = parser.parse_args()

    # Load test questions
    questions = load_mentalchat_test_set(args.num_questions)

    # Load Mistral model (used for all pipeline configurations)
    print("\nLoading Mistral-7B-Instruct...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Mistral model loaded!")

    # Store all results
    all_results = {}

    # 1. Base Mistral
    all_results["Base Mistral-7B"] = evaluate_base_mistral(questions, model, tokenizer)

    # 2. Full pipeline
    all_results["Full 3-Component Pipeline"] = evaluate_full_pipeline(questions, model, tokenizer)

    # 3. No clinical
    all_results["No Clinical Pipeline"] = evaluate_no_clinical(questions, model, tokenizer)

    # 4. No opinion
    all_results["No Opinion Pipeline"] = evaluate_no_opinion(questions, model, tokenizer)

    # Clean up Mistral model before loading finetuned
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # 5. Finetuned MentalChat
    all_results["Finetuned MentalChat-16K"] = evaluate_finetuned_mentalchat(questions)

    # Print summary
    print_summary(all_results)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
