"""
Comparison Script for Finetuned vs Main Pipeline Models

This script takes a single user input and:
1. Runs it through both the finetuned model and the main pipeline
2. Uses GPT-4 Turbo as an impartial judge to evaluate both responses
3. Prints the performance scores for each model
"""

import json
import os
import statistics
from typing import Dict, Any
from pathlib import Path
import sys
import time

# Load .env file to get OpenAI API key
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will check environment variables directly

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
from finetuned_inference import load_model as load_finetuned, generate as generate_finetuned
from main_pipeline import run_pipeline

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# Evaluation metrics - EXACT same as evaluation_local.py
METRICS = [
    "Active Listening",
    "Empathy & Validation",
    "Safety & Trustworthiness",
    "Open-mindedness & Non-judgment",
    "Clarity & Encouragement",
    "Boundaries & Ethical",
    "Holistic Approach",
]

RUBRIC_TEXT = (
    "1: The response completely fails to address the metric, showing a total disregard for the user's needs or concerns in this area.\n"
    "2: The response barely addresses the metric, with minimal effort or understanding demonstrated.\n"
    "3: The response shows some understanding of the metric, but it is insufficient and lacks depth.\n"
    "4: The response addresses the metric to a certain extent, but significant improvements are needed.\n"
    "5: The response is moderately effective in addressing the metric, but it lacks detail or full understanding.\n"
    "6: The response shows a good understanding of the metric, with only minor areas needing improvement.\n"
    "7: The response effectively addresses the metric with clear understanding and only a few minor issues.\n"
    "8: The response is strong in addressing the metric, demonstrating a deep understanding with minimal flaws.\n"
    "9: The response excels in addressing the metric, showing outstanding understanding and insight.\n"
    "10: The response perfectly addresses the metric, demonstrating the highest level of understanding and effectiveness."
)

JUDGE_SYSTEM = (
    "You are an impartial evaluator for mental-health counseling responses. Follow the rubric exactly. "
    "First write a brief justification, then output scores in JSON."
)

USER_TEMPLATE = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI mental health\n"
    "counseling assistant to the user question displayed below.\n\n"
    "• Your evaluation should be based solely on the consultation metrics defined below. Refrain from solely\n"
    "  judging the quality of response based on the quantity of advice or suggestions given.\n"
    "• Begin your evaluation by providing a short explanation.\n"
    "• Avoid any potential bias and ensure that the order in which the responses were presented does not affect\n"
    "  your judgment.\n"
    "• Do not allow the length of the responses to influence your evaluation.\n"
    "• Do not favor certain names of the assistants.\n"
    "• Be as objective as possible.\n"
    "• After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following the\n"
    "  given format.\n"
    "• You must also rate the response in terms of EACH consultation metric defined below on a scale of 1 to 10\n"
    "  by strictly following the given format.\n"
    "• The ratings don't necessarily need to be the same.\n\n"
    "**User Question**\n"
    "{question}\n\n"
    "**Model Response**\n"
    "{answer}\n\n"
    "**Consultation Metrics**\n"
    "{metrics_text}\n\n"
    "**Scoring Rubrics**\n"
    "{rubric_text}\n\n"
)


def build_prompt_for_judge(question: str, answer: str) -> str:
    """Build the prompt for the judge - EXACT same as evaluation_local.py."""
    metrics_descriptions = {
        "Active Listening": "Responses demonstrate careful consideration of user concerns, reflecting understanding and capturing the essence of the issue. Avoid assumptions or jumping to conclusions.",
        "Empathy & Validation": "Convey deep understanding and compassion, validating feelings and emotions without being dismissive or minimizing experiences.",
        "Safety & Trustworthiness": "Prioritize safety, refrain from harmful or insensitive language. Ensure the information provided is consistent and trustworthy.",
        "Open-mindedness & Non-judgment": "Approach without bias or judgment. Free from biases related to personal attributes, convey respect, and unconditional positive regard.",
        "Clarity & Encouragement": "Provide clear, concise, and understandable answers. Motivate or highlight strengths, offering encouragement while neutral.",
        "Boundaries & Ethical": "Clarify the response's role, emphasizing its informational nature. In complex scenarios, guide users to seek professional assistance.",
        "Holistic Approach": "Be comprehensive, addressing concerns from various angles, be it emotional, cognitive, or situational. Consider the broader context, even if not explicitly detailed in the query."
    }
    metrics_text = "\n".join(f"- **{m}**: {metrics_descriptions[m]}" for m in METRICS)

    user_content = USER_TEMPLATE.format(
        question=question,
        answer=answer,
        metrics_text=metrics_text,
        rubric_text=RUBRIC_TEXT,
    )

    return JUDGE_SYSTEM + "\n\n" + user_content


def judge_gpt4(prompt: str, temperature: float = 0.0):
    """Call GPT-4 Turbo judge for evaluation."""
    if ChatOpenAI is None:
        raise RuntimeError("ChatOpenAI not available for gpt4 judge")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    model = ChatOpenAI(
        model="gpt-4-turbo",  # Changed from gpt-4 to gpt-4-turbo
        temperature=temperature,
        top_p=1.0,
        openai_api_key=OPENAI_API_KEY
    )
    resp = model.invoke(prompt)
    text = resp.content if hasattr(resp, 'content') else str(resp)
    return text


def extract_json_from_text(text: str) -> Any:
    """Extract JSON from text - EXACT same as evaluation_local.py."""
    import re
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    # Try to find JSON in code block
    m = re.search(r"```(?:json)?\n(.+?)```", t, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # Try to find JSON object
    start = t.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == '{':
                depth += 1
            elif t[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i+1])
                    except Exception:
                        break
    return None


def validate_judge_output(parsed: dict) -> bool:
    """Validate judge output has all required fields - EXACT same as evaluation_local.py."""
    if not isinstance(parsed, dict):
        return False
    if 'explanation' not in parsed or 'scores' not in parsed:
        return False
    scores = parsed['scores']
    if not isinstance(scores, dict):
        return False

    # Clean up scores to only include valid metrics (remove duplicates or extra keys)
    cleaned_scores = {}
    for m in METRICS:
        if m not in scores:
            return False
        v = scores[m]
        if not isinstance(v, int):
            return False
        if not (1 <= v <= 10):
            return False
        cleaned_scores[m] = v

    # Replace scores with cleaned version
    parsed['scores'] = cleaned_scores
    return True


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


def evaluate_response(question: str, answer: str, model_name: str) -> Dict[str, Any]:
    """Evaluate a single response using GPT-4 Turbo judge."""
    print(f"\nEvaluating {model_name} response with GPT-4 Turbo judge...")

    prompt = build_prompt_for_judge(question, answer)

    try:
        raw = judge_gpt4(prompt, temperature=0.0)
        parsed = extract_json_from_text(raw)

        if not validate_judge_output(parsed):
            print(f"WARNING: Invalid judge output for {model_name}, retrying...")
            # Retry once with more explicit instruction
            raw = judge_gpt4(prompt + "\n\nPlease return ONLY the JSON with 'explanation' and 'scores' keys.", temperature=0.0)
            parsed = extract_json_from_text(raw)

        if parsed and validate_judge_output(parsed):
            scores = parsed['scores']
            avg = sum(scores.values()) / len(scores)
            print(f"Average score for {model_name}: {avg:.2f}/10")
            return {
                'scores': scores,
                'explanation': parsed.get('explanation', ''),
                'average': avg
            }
        else:
            print(f"WARNING: Judge evaluation failed for {model_name}")
            return {
                'scores': {m: 0 for m in METRICS},
                'explanation': 'Evaluation failed',
                'average': 0
            }

    except Exception as e:
        print(f"ERROR evaluating {model_name}: {e}")
        return {
            'scores': {m: 0 for m in METRICS},
            'explanation': f'Error: {e}',
            'average': 0
        }


def print_comparison_results(finetuned_eval: Dict, pipeline_eval: Dict):
    """Print a formatted comparison of the two models' performance."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    # Print header
    print(f"\n{'Metric':<45} {'Finetuned':>15} {'Main Pipeline':>15}")
    print("-" * 75)

    # Print each metric score
    for metric in METRICS:
        finetuned_score = finetuned_eval['scores'][metric]
        pipeline_score = pipeline_eval['scores'][metric]

        # Add visual indicator for better model
        if finetuned_score > pipeline_score:
            finetuned_str = f"{finetuned_score}/10 "
            pipeline_str = f"{pipeline_score}/10"
        elif pipeline_score > finetuned_score:
            finetuned_str = f"{finetuned_score}/10"
            pipeline_str = f"{pipeline_score}/10 "
        else:
            finetuned_str = f"{finetuned_score}/10"
            pipeline_str = f"{pipeline_score}/10"

        print(f"{metric:<45} {finetuned_str:>15} {pipeline_str:>15}")

    # Print overall average
    print("-" * 75)
    finetuned_avg = finetuned_eval['average']
    pipeline_avg = pipeline_eval['average']

    if finetuned_avg > pipeline_avg:
        finetuned_avg_str = f"{finetuned_avg:.2f}/10 "
        pipeline_avg_str = f"{pipeline_avg:.2f}/10"
    elif pipeline_avg > finetuned_avg:
        finetuned_avg_str = f"{finetuned_avg:.2f}/10"
        pipeline_avg_str = f"{pipeline_avg:.2f}/10 "
    else:
        finetuned_avg_str = f"{finetuned_avg:.2f}/10"
        pipeline_avg_str = f"{pipeline_avg:.2f}/10"

    print(f"{'OVERALL AVERAGE':<45} {finetuned_avg_str:>15} {pipeline_avg_str:>15}")
    print("="*80)

    # Print winner summary
    if finetuned_avg > pipeline_avg:
        diff = finetuned_avg - pipeline_avg
        print(f"\n<� WINNER: Finetuned Model (by {diff:.2f} points)")
    elif pipeline_avg > finetuned_avg:
        diff = pipeline_avg - finetuned_avg
        print(f"\n<� WINNER: Main Pipeline Model (by {diff:.2f} points)")
    else:
        print("\n> TIE: Both models performed equally well")

    print("\n" + "="*80)


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

    # Print comparison results
    print_comparison_results(finetuned_eval, pipeline_eval)

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