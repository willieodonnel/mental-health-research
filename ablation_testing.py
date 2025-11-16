"""
Ablation Testing Script - Compare Full Pipeline vs 2-Component Pipeline

This script compares:
1. Full 3-component pipeline (Clinical Description -> Professional Opinion -> Response)
2. Ablated 2-component pipeline (Input -> Professional Opinion -> Response)

Both use the EXACT same prompts but the ablated version skips the clinical description step.
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
from judging import (
    evaluate_response,
    print_comparison_results
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


def generate(model, tokenizer, prompt, max_length=1024):
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
            max_length=max_length,
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
    clinical_prompt = f"""Convert this to third-person clinical language:
"{user_input}"

Change "I" to "The patient", keep it concise and clinical."""

    clinical_description = generate(model, tokenizer, clinical_prompt)
    print("\n1. Clinical Description:")
    print(clinical_description)

    # Component 2: Professional opinion
    opinion_prompt = f"""As a mental health professional, provide a brief assessment of:
{clinical_description}

Identify key concerns and provide professional opinion."""

    professional_opinion = generate(model, tokenizer, opinion_prompt)
    print("\n2. Professional Opinion:")
    print(professional_opinion)

    # Component 3: Final response - EXACT same prompt as main_pipeline.py
    response_prompt = f"""You are an empathetic counselor. Using the professional context below, respond helpfully to the patient's concern.

Original concern: {user_input}

Professional context: {professional_opinion}

Provide a compassionate, helpful response:"""

    final_response = generate(model, tokenizer, response_prompt)
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


def run_ablated_pipeline(model, tokenizer, user_input):
    """
    Run the ABLATED 2-component pipeline on user input.
    Components: Input -> Professional Opinion -> Response
    (Skips the clinical description step)
    """
    print("\n" + "="*60)
    print("RUNNING ABLATED 2-COMPONENT PIPELINE")
    print("="*60)

    start_time = time.time()

    # Component 1 (was 2): Professional opinion - directly from user input
    # Using user input instead of clinical description
    opinion_prompt = f"""As a mental health professional, provide a brief assessment of:
{user_input}

Identify key concerns and provide professional opinion."""

    professional_opinion = generate(model, tokenizer, opinion_prompt)
    print("\n1. Professional Opinion (Direct from Input):")
    print(professional_opinion)

    # Component 2 (was 3): Final response - EXACT same prompt structure
    response_prompt = f"""You are an empathetic counselor. Using the professional context below, respond helpfully to the patient's concern.

Original concern: {user_input}

Professional context: {professional_opinion}

Provide a compassionate, helpful response:"""

    final_response = generate(model, tokenizer, response_prompt)
    print("\n2. Final Response:")
    print(final_response)

    generation_time = time.time() - start_time

    return {
        "user_input": user_input,
        "professional_opinion": professional_opinion,
        "final_response": final_response,
        "generation_time": generation_time,
        "pipeline_type": "ablated_2_components"
    }


def ablation_test(user_input: str):
    """
    Main ablation testing function that runs both pipeline versions and evaluates them.

    Args:
        user_input: The mental health question/concern from the user
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: Full vs Ablated Pipeline")
    print("="*80)
    print(f"\nUser Input: {user_input}")

    # Load model once for both pipelines
    model, tokenizer = load_model()

    # Run full 3-component pipeline
    full_result = run_full_pipeline(model, tokenizer, user_input)

    # Run ablated 2-component pipeline
    ablated_result = run_ablated_pipeline(model, tokenizer, user_input)

    # Evaluate both responses
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    full_eval = evaluate_response(
        user_input,
        full_result['final_response'],
        "Full 3-Component Pipeline"
    )

    ablated_eval = evaluate_response(
        user_input,
        ablated_result['final_response'],
        "Ablated 2-Component Pipeline"
    )

    # Print comparison results using centralized function
    print_comparison_results(
        full_eval,
        ablated_eval,
        "Full (3-comp)",
        "Ablated (2-comp)",
        "ABLATION STUDY RESULTS"
    )

    # Print ablation-specific insights
    if full_eval['average'] > ablated_eval['average']:
        print("\nThe clinical description step adds value to the pipeline!")
    elif ablated_eval['average'] > full_eval['average']:
        print("The clinical description step may not be necessary!")
    else:
        print("The clinical description step has neutral impact.")

    return {
        'user_input': user_input,
        'full_pipeline': {
            'result': full_result,
            'evaluation': full_eval
        },
        'ablated_pipeline': {
            'result': ablated_result,
            'evaluation': ablated_eval
        }
    }


def main():
    """Main function to run the ablation study."""
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Study: Full vs Ablated Pipeline')
    parser.add_argument('--input', type=str,
                       help='User input to test (optional, will prompt if not provided)')

    args = parser.parse_args()

    if args.input:
        user_input = args.input
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("ABLATION STUDY TOOL")
        print("="*80)
        print("\nThis tool will:")
        print("1. Run your input through the FULL 3-component pipeline")
        print("   (Clinical Description -> Professional Opinion -> Response)")
        print("2. Run your input through the ABLATED 2-component pipeline")
        print("   (Input -> Professional Opinion -> Response)")
        print("3. Use GPT-4 Turbo to evaluate both responses")
        print("4. Show you which pipeline performs better")
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