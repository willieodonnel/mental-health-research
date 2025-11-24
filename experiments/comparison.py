"""
Comparison Script for Finetuned vs Main Pipeline Models

This script takes a single user input and:
1. Runs it through both the finetuned model and the main pipeline
2. Uses GPT-4 Turbo as an impartial judge to evaluate both responses
3. Prints the performance scores for each model
"""

import sys
import time
from typing import Dict, Any
from pathlib import Path

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
from finetuned_mentalchat_model import load_model as load_finetuned, generate as generate_finetuned
from main_pipeline import run_pipeline

# Import centralized judging functions
from judging import (
    evaluate_response,
    print_comparison_results
)


def run_finetuned_model(user_input: str) -> Dict[str, Any]:
    """Run the finetuned model and return the response."""
    print("\n" + "="*60)
    print("RUNNING FINETUNED MODEL (Llama-3.2-1B-Instruct)")
    print("="*60)

    start_time = time.time()

    # Load and run finetuned model
    model, tokenizer = load_finetuned()
    response_data = generate_finetuned(model, tokenizer, user_input)

    # Extract the actual response text from the pipeline output
    if isinstance(response_data, list) and len(response_data) > 0:
        response_text = response_data[0]['generated_text'][-1]['content']
    else:
        response_text = str(response_data)

    generation_time = time.time() - start_time

    print(f"\nFinetuned Response ({generation_time:.1f}s):")
    print(response_text)

    return {
        'response': response_text,
        'generation_time': generation_time,
        'model': 'finetuned_llama_3.2_1b'
    }


def run_main_pipeline_model(user_input: str) -> Dict[str, Any]:
    """Run the main pipeline model and return the response."""
    print("\n" + "="*60)
    print("RUNNING MAIN PIPELINE MODEL (Mistral-7B with 3 Components)")
    print("="*60)

    start_time = time.time()

    # Run the pipeline
    result = run_pipeline(user_input)

    generation_time = time.time() - start_time

    # The final response is what we'll evaluate
    response_text = result['final_response']

    print(f"\nMain Pipeline Response ({generation_time:.1f}s):")
    print(response_text)

    return {
        'response': response_text,
        'generation_time': generation_time,
        'model': 'main_pipeline_mistral_7b',
        'full_result': result  # Keep all components for reference
    }


def compare_models(user_input: str):
    """
    Main comparison function that runs both models and evaluates them.

    Args:
        user_input: The mental health question/concern from the user
    """
    print("\n" + "="*80)
    print("MENTAL HEALTH MODEL COMPARISON")
    print("="*80)
    print(f"\nUser Input: {user_input}")

    # Run finetuned model
    finetuned_result = run_finetuned_model(user_input)

    # Run main pipeline model
    pipeline_result = run_main_pipeline_model(user_input)

    # Evaluate both responses
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    finetuned_eval = evaluate_response(
        user_input,
        finetuned_result['response'],
        "Finetuned Model"
    )

    pipeline_eval = evaluate_response(
        user_input,
        pipeline_result['response'],
        "Main Pipeline Model"
    )

    # Print comparison results using the centralized function
    print_comparison_results(
        finetuned_eval,
        pipeline_eval,
        "Finetuned",
        "Main Pipeline",
        "COMPARISON RESULTS"
    )

    return {
        'user_input': user_input,
        'finetuned': {
            'response': finetuned_result['response'],
            'generation_time': finetuned_result['generation_time'],
            'evaluation': finetuned_eval
        },
        'main_pipeline': {
            'response': pipeline_result['response'],
            'generation_time': pipeline_result['generation_time'],
            'evaluation': pipeline_eval
        }
    }


def main():
    """Main function to run the comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare Finetuned vs Main Pipeline Models')
    parser.add_argument('--input', type=str,
                       help='User input to test (optional, will prompt if not provided)')

    args = parser.parse_args()

    if args.input:
        user_input = args.input
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("MENTAL HEALTH MODEL COMPARISON TOOL")
        print("="*80)
        print("\nThis tool will:")
        print("1. Run your input through both the finetuned model and main pipeline")
        print("2. Use GPT-4 Turbo to evaluate both responses")
        print("3. Show you a detailed performance comparison")
        print("\nEnter 'quit' to exit\n")

        while True:
            user_input = input("Enter your mental health concern/question:\n> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_input:
                print("Please enter a valid input.\n")
                continue

            # Run comparison
            result = compare_models(user_input)

            # Ask if they want to try another
            print("\nWould you like to compare another input? (yes/no)")
            if input("> ").strip().lower() not in ['yes', 'y']:
                print("\nGoodbye!")
                break

    # If single input provided via argument
    if args.input:
        compare_models(user_input)


if __name__ == "__main__":
    main()