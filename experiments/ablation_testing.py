"""
Ablation Testing Script - Compare Three Pipeline Variants

This script compares:
1. Full 3-component pipeline (Clinical Description -> Professional Opinion -> Response)
2. No Clinical pipeline (Input -> Professional Opinion -> Response)
3. No Professional Opinion pipeline (Clinical Description -> Response)

All use the EXACT same prompts for their respective components from pipeline_pieces.py.
"""

import time
from typing import Dict, Any
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import evaluation components
sys.path.insert(0, str(Path(__file__).parent))

# Import centralized judging functions
from evaluation.judging import (
    evaluate_response,
    print_comparison_results,
    METRICS
)

# Import pipeline components
from src.models.pipeline_pieces import (
    run_clinical_description,
    run_professional_opinion,
    run_final_response
)


def load_model():
    """Load Mistral model once for all components."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print("Loading Mistral model for ablation testing...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Model loaded successfully!")
    return model, tokenizer


def generate(model, tokenizer, prompt):
    """Generate response using Mistral."""
    # Format in Mistral instruction format
    formatted_prompt = f"[INST] {prompt} [/INST]"

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract response after [/INST]
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


def run_full_pipeline(model, tokenizer, user_input):
    """
    Run the FULL 3-component pipeline on user input.
    Components: Clinical Description -> Professional Opinion -> Response
    """
    print("\n" + "="*60)
    print("RUNNING FULL 3-COMPONENT PIPELINE")
    print("="*60)

    start_time = time.time()

    # Component 1: Convert to third-person clinical language
    clinical_description = run_clinical_description(model, tokenizer, generate, user_input)
    print("\n1. Clinical Description:")
    print(clinical_description)

    # Component 2: Professional opinion from clinical description
    professional_opinion = run_professional_opinion(model, tokenizer, generate, clinical_description)
    print("\n2. Professional Opinion:")
    print(professional_opinion)

    # Component 3: Final response using professional context
    final_response = run_final_response(model, tokenizer, generate, user_input, professional_opinion, "professional")
    print("\n3. Final Response:")
    print(final_response)

    generation_time = time.time() - start_time

    return {
        "user_input": user_input,
        "clinical_description": clinical_description,
        "professional_opinion": professional_opinion,
        "final_response": final_response,
        "generation_time": generation_time,
        "pipeline_type": "full_3_components"
    }


def run_no_clinical_pipeline(model, tokenizer, user_input):
    """
    Run the NO CLINICAL 2-component pipeline on user input.
    Components: Input -> Professional Opinion -> Response
    (Skips the clinical description step)
    """
    print("\n" + "="*60)
    print("RUNNING NO CLINICAL PIPELINE (2 Components)")
    print("="*60)

    start_time = time.time()

    # Component 1: Professional opinion directly from user input
    professional_opinion = run_professional_opinion(model, tokenizer, generate, user_input)
    print("\n1. Professional Opinion (Direct from Input):")
    print(professional_opinion)

    # Component 2: Final response using professional context
    final_response = run_final_response(model, tokenizer, generate, user_input, professional_opinion, "professional")
    print("\n2. Final Response:")
    print(final_response)

    generation_time = time.time() - start_time

    return {
        "user_input": user_input,
        "professional_opinion": professional_opinion,
        "final_response": final_response,
        "generation_time": generation_time,
        "pipeline_type": "no_clinical_2_components"
    }


def run_no_opinion_pipeline(model, tokenizer, user_input):
    """
    Run the NO PROFESSIONAL OPINION 2-component pipeline on user input.
    Components: Clinical Description -> Response
    (Skips the professional opinion step)
    """
    print("\n" + "="*60)
    print("RUNNING NO PROFESSIONAL OPINION PIPELINE (2 Components)")
    print("="*60)

    start_time = time.time()

    # Component 1: Convert to third-person clinical language
    clinical_description = run_clinical_description(model, tokenizer, generate, user_input)
    print("\n1. Clinical Description:")
    print(clinical_description)

    # Component 2: Final response using clinical context
    final_response = run_final_response(model, tokenizer, generate, user_input, clinical_description, "clinical")
    print("\n2. Final Response:")
    print(final_response)

    generation_time = time.time() - start_time

    return {
        "user_input": user_input,
        "clinical_description": clinical_description,
        "final_response": final_response,
        "generation_time": generation_time,
        "pipeline_type": "no_opinion_2_components"
    }


def print_three_way_comparison(
    eval1: Dict[str, Any],
    eval2: Dict[str, Any],
    eval3: Dict[str, Any]
) -> None:
    """
    Print a formatted comparison of three pipeline evaluations.

    Args:
        eval1: Full pipeline evaluation results
        eval2: No Clinical pipeline evaluation results
        eval3: No Opinion pipeline evaluation results
    """
    print("\n" + "="*90)
    print("THREE-WAY ABLATION COMPARISON")
    print("="*90)

    # Define names
    name1 = "Full (3-comp)"
    name2 = "No Clinical"
    name3 = "No Opinion"

    # Print header
    print(f"\n{'Metric':<35} {name1:>15} {name2:>15} {name3:>15}")
    print("-" * 90)

    # Print each metric score
    for metric in METRICS:
        score1 = eval1['scores'][metric]
        score2 = eval2['scores'][metric]
        score3 = eval3['scores'][metric]

        # Find the best score
        max_score = max(score1, score2, score3)

        # Add checkmarks to best score(s)
        str1 = f"{score1}/10 ✓" if score1 == max_score else f"{score1}/10"
        str2 = f"{score2}/10 ✓" if score2 == max_score else f"{score2}/10"
        str3 = f"{score3}/10 ✓" if score3 == max_score else f"{score3}/10"

        print(f"{metric:<35} {str1:>15} {str2:>15} {str3:>15}")

    # Print overall average
    print("-" * 90)
    avg1 = eval1['average']
    avg2 = eval2['average']
    avg3 = eval3['average']

    max_avg = max(avg1, avg2, avg3)

    avg1_str = f"{avg1:.2f}/10 ✓" if avg1 == max_avg else f"{avg1:.2f}/10"
    avg2_str = f"{avg2:.2f}/10 ✓" if avg2 == max_avg else f"{avg2:.2f}/10"
    avg3_str = f"{avg3:.2f}/10 ✓" if avg3 == max_avg else f"{avg3:.2f}/10"

    print(f"{'OVERALL AVERAGE':<35} {avg1_str:>15} {avg2_str:>15} {avg3_str:>15}")
    print("="*90)

    # Print winner summary
    scores_dict = {
        name1: avg1,
        name2: avg2,
        name3: avg3
    }

    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    winner_name, winner_score = sorted_scores[0]

    # Check for ties
    winners = [name for name, score in sorted_scores if score == winner_score]

    if len(winners) > 1:
        print(f"\nTIE: {' and '.join(winners)} (all scored {winner_score:.2f}/10)")
    else:
        second_score = sorted_scores[1][1]
        diff = winner_score - second_score
        print(f"\nWINNER: {winner_name} (scored {winner_score:.2f}/10, {diff:.2f} points ahead)")

    # Print ablation insights
    print("\n" + "="*90)
    print("ABLATION INSIGHTS")
    print("="*90)

    clinical_impact = avg1 - avg3  # Full vs No Opinion (difference shows clinical step value)
    opinion_impact = avg1 - avg2   # Full vs No Clinical (difference shows opinion step value)

    print(f"\nClinical Description Impact: {clinical_impact:+.2f} points")
    if clinical_impact > 0.5:
        print("  → Clinical description step adds significant value")
    elif clinical_impact < -0.5:
        print("  → Clinical description step may hurt performance")
    else:
        print("  → Clinical description step has minimal impact")

    print(f"\nProfessional Opinion Impact: {opinion_impact:+.2f} points")
    if opinion_impact > 0.5:
        print("  → Professional opinion step adds significant value")
    elif opinion_impact < -0.5:
        print("  → Professional opinion step may hurt performance")
    else:
        print("  → Professional opinion step has minimal impact")

    print("\n" + "="*90)


def ablation_test(user_input: str):
    """
    Main ablation testing function that runs all three pipeline versions and evaluates them.

    Args:
        user_input: The mental health question/concern from the user
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: Comparing Three Pipeline Variants")
    print("="*80)
    print(f"\nUser Input: {user_input}")

    # Load model once for all pipelines
    model, tokenizer = load_model()

    # Run all three pipeline variants
    full_result = run_full_pipeline(model, tokenizer, user_input)
    no_clinical_result = run_no_clinical_pipeline(model, tokenizer, user_input)
    no_opinion_result = run_no_opinion_pipeline(model, tokenizer, user_input)

    # Evaluate all three responses
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    full_eval = evaluate_response(
        user_input,
        full_result['final_response'],
        "Full 3-Component Pipeline"
    )

    no_clinical_eval = evaluate_response(
        user_input,
        no_clinical_result['final_response'],
        "No Clinical Pipeline"
    )

    no_opinion_eval = evaluate_response(
        user_input,
        no_opinion_result['final_response'],
        "No Professional Opinion Pipeline"
    )

    # Print three-way comparison
    print_three_way_comparison(full_eval, no_clinical_eval, no_opinion_eval)

    return {
        'user_input': user_input,
        'full_pipeline': {
            'result': full_result,
            'evaluation': full_eval
        },
        'no_clinical_pipeline': {
            'result': no_clinical_result,
            'evaluation': no_clinical_eval
        },
        'no_opinion_pipeline': {
            'result': no_opinion_result,
            'evaluation': no_opinion_eval
        }
    }


def main():
    """Main function to run the ablation study."""
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Study: Three-Way Pipeline Comparison')
    parser.add_argument('--input', type=str,
                       help='User input to test (optional, will prompt if not provided)')

    args = parser.parse_args()

    if args.input:
        user_input = args.input
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("ABLATION STUDY TOOL - THREE-WAY COMPARISON")
        print("="*80)
        print("\nThis tool will:")
        print("1. Run your input through the FULL 3-component pipeline")
        print("   (Clinical Description -> Professional Opinion -> Response)")
        print("2. Run your input through the NO CLINICAL 2-component pipeline")
        print("   (Input -> Professional Opinion -> Response)")
        print("3. Run your input through the NO PROFESSIONAL OPINION 2-component pipeline")
        print("   (Clinical Description -> Response)")
        print("4. Use GPT-4 Turbo to evaluate all three responses")
        print("5. Show you which pipeline performs best")
        print("\nEnter 'quit' to exit\n")

        while True:
            user_input = input("Enter your mental health concern/question:\n> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_input:
                print("Please enter a valid input.\n")
                continue

            # Run ablation test
            result = ablation_test(user_input)

            # Ask if they want to try another
            print("\nWould you like to test another input? (yes/no)")
            if input("> ").strip().lower() not in ['yes', 'y']:
                print("\nGoodbye!")
                break

    # If single input provided via argument
    if args.input:
        ablation_test(user_input)


if __name__ == "__main__":
    main()
