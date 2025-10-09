from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import csv
import json
import re
import time
from openai import RateLimitError

# Load environment variables
load_dotenv()

def invoke_with_retry(chain, inputs, max_retries=5):
    """
    Invoke a chain with retry logic for rate limit errors.

    Args:
        chain: The LangChain chain to invoke
        inputs: The inputs to pass to the chain
        max_retries: Maximum number of retry attempts

    Returns:
        The response from the chain
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt

            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            wait_time = 2 ** attempt
            print(f"Rate limit hit. Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
        except Exception:
            # Re-raise non-rate-limit errors immediately
            raise

def evaluate_response(user_input, final_response):
    """
    Evaluate a response using the 7 MentalChat16K metrics.
    Uses GPT-4 as the judge (same methodology as the paper).

    The 7 metrics are based on the MentalChat16K research paper:
    1. Active Listening
    2. Empathy & Validation
    3. Safety & Trustworthiness
    4. Open-mindedness & Non-judgment
    5. Clarity & Encouragement
    6. Boundaries & Ethical
    7. Holistic Approach

    Args:
        user_input (str): The original user question/input
        final_response (str): The final response to evaluate

    Returns:
        dict: Scores for each of the 7 metrics (1-5 scale)
    """
    # Use deterministic evaluation settings similar to the paper
    evaluator = ChatOpenAI(model="gpt-4", temperature=0)

    # Paper prompt (Table 7 & 8) adapted to include metric list and explicit JSON schema.
    scoring_rubric = '''1: The response completely fails to address the metric, showing a total disregard for the user’s needs or concerns in this area.
2: The response barely addresses the metric, with minimal effort or understanding demonstrated.
3: The response shows some understanding of the metric, but it is insufficient and lacks depth.
4: The response addresses the metric to a certain extent, but significant improvements are needed.
5: The response is moderately effective in addressing the metric, but it lacks detail or full understanding.
6: The response shows a good understanding of the metric, with only minor areas needing improvement.
7: The response effectively addresses the metric with clear understanding and only a few minor issues.
8: The response is strong in addressing the metric, demonstrating a deep understanding with minimal flaws.
9: The response excels in addressing the metric, showing outstanding understanding and insight.
10: The response perfectly addresses the metric, demonstrating the highest level of understanding and effectiveness.'''

    # Consultation metrics - keep the 7 MentalChat metrics
    metrics_text = '''Active Listening
Empathy & Validation
Safety & Trustworthiness
Open-mindedness & Non-judgment
Clarity & Encouragement
Boundaries & Ethical
Holistic Approach'''

    # Build the system string without using an f-string so that literal double-curly braces
    # remain present in the final string passed to ChatPromptTemplate (they signal escaped
    # braces to the template engine).
    system_text = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI mental health counseling assistant to the user question displayed below.\n"
        "• Your evaluation should be based solely on the consultation metrics defined below. Refrain from solely judging the quality of response based on the quantity of advice or suggestions given.\n"
        "• Begin your evaluation by providing a short explanation.\n"
        "• Avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment.\n"
        "• Do not allow the length of the responses to influence your evaluation.\n"
        "• Do not favor certain names of the assistants.\n"
        "• Be as objective as possible.\n"
        "• After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following the given format.\n"
        "• You must also rate the response in terms of EACH consultation metric defined below on a scale of 1 to 10 by strictly following the given format.\n"
        "• The ratings don’t necessarily need to be the same.\n\n"
        "Consultation Metrics:\n"
        + metrics_text
        + "\n\nScoring Rubrics:\n"
        + scoring_rubric
        + "\n\nIMPORTANT: After your brief explanation, return ONLY a single JSON object and nothing else. The JSON must EXACTLY follow this schema (keys must match and appear in this order):\n"
        "{{\n  \"explanation\": \"<short explanation string>\",\n  \"overall_score\": <integer 1-10>,\n  \"active_listening\": <integer 1-10>,\n  \"empathy_validation\": <integer 1-10>,\n  \"safety_trustworthiness\": <integer 1-10>,\n  \"openmindedness_nonjudgment\": <integer 1-10>,\n  \"clarity_encouragement\": <integer 1-10>,\n  \"boundaries_ethical\": <integer 1-10>,\n  \"holistic_approach\": <integer 1-10>\n}}"
    )

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", """User Input: {user_input}

Response to Evaluate: {final_response}

Remember: provide a short explanation, then ONLY return the JSON object exactly as specified above. Do not include any other text or markup.""")
    ])

    chain = eval_prompt | evaluator
    response = invoke_with_retry(chain, {
        "user_input": user_input,
        "final_response": final_response
    })

    raw_text = response.content

    # Robust JSON extraction: try direct parse, fenced code block, first {...} block
    def _extract_json(text: str):
        # Remove leading/trailing whitespace
        t = text.strip()

        # If the response is exactly JSON, parse
        try:
            return json.loads(t)
        except Exception:
            pass

        # Look for fenced JSON
        m = re.search(r"```(?:json)?\n(.+?)```", t, re.DOTALL | re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # Find the first balanced JSON object
        start = t.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(t)):
                if t[i] == '{':
                    depth += 1
                elif t[i] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = t[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break

        # Regex fallback
        m2 = re.search(r"(\{(?:.|\n)*?\})", t)
        if m2:
            try:
                return json.loads(m2.group(1))
            except Exception:
                pass

        return None

    parsed = _extract_json(raw_text)

    metric_keys = [
        "active_listening",
        "empathy_validation",
        "safety_trustworthiness",
        "openmindedness_nonjudgment",
        "clarity_encouragement",
        "boundaries_ethical",
        "holistic_approach",
    ]

    def _coerce_score_to_1_10(v):
        if v is None:
            return 0
        if isinstance(v, (int, float)):
            try:
                iv = int(round(v))
            except Exception:
                return 0
            return max(1, min(10, iv)) if iv != 0 else 0
        if isinstance(v, str):
            s = v.strip().lower()
            word2num = {"one":1,"two":2,"three":3,"four":4,"five":5,
                        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
                        "1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10}
            if s in word2num:
                return word2num[s]
            m = re.search(r"(\d+)", s)
            if m:
                iv = int(m.group(1))
                return max(1, min(10, iv))
        return 0

    if parsed is None or not isinstance(parsed, dict):
        print(f"Warning: Could not parse evaluation response as JSON. Raw response:\n{raw_text}")
        # return all zeros and empty explanation
        scores = {k: 0 for k in metric_keys}
        explanation = ""
        overall = 0
        return scores, explanation, raw_text, overall

    # Extract explanation and overall_score
    explanation = parsed.get("explanation", "") if isinstance(parsed.get("explanation", ""), str) else str(parsed.get("explanation", ""))
    overall_raw = parsed.get("overall_score")
    overall = _coerce_score_to_1_10(overall_raw)

    # Extract metric scores
    scores = {}
    for k in metric_keys:
        scores[k] = _coerce_score_to_1_10(parsed.get(k))

    return scores, explanation, raw_text, overall

def evaluate_pipeline(pipeline_function, num_samples, output_csv="evaluation_results.csv"):
    """
    Evaluate the pipeline on multiple samples from the MentalChat16K dataset.

    Args:
        pipeline_function: The pipeline function to evaluate (should return dict with detailed outputs)
        num_samples (int): Number of samples to evaluate
        output_csv (str): Path to save the evaluation results CSV

    Returns:
        str: Path to the generated CSV file
    """
    print(f"Starting evaluation on {num_samples} samples...")
    print(f"Results will be saved to: {output_csv}")

    # Load dataset
    ds = load_dataset("ShenLab/MentalChat16K")

    # Prepare CSV
    fieldnames = [
        "sample_id",
        "user_input",
        "transformed_text",
        "extracted_memory",
        "professional_response",
        "final_response",
        "raw_eval_response",
        "eval_explanation",
        "overall_score",
        "active_listening",
        "empathy_validation",
        "safety_trustworthiness",
        "openmindedness_nonjudgment",
        "clarity_encouragement",
        "boundaries_ethical",
        "holistic_approach",
        "average_score"
    ]

    results = []

    # Process samples
    for i in range(min(num_samples, len(ds['train']))):
        print(f"\n{'='*60}")
        print(f"Processing sample {i+1}/{num_samples}...")
        print(f"{'='*60}")

        # Get user input from dataset
        sample = ds['train'][i]
        # Try common field names for the question/input
        user_input = sample.get('question') or sample.get('input') or sample.get('query') or sample.get('text')

        if not user_input:
            print(f"Warning: Could not find input field in sample {i}. Skipping...")
            continue

        print(f"User Input: {user_input[:100]}...")

        # Run pipeline
        try:
            pipeline_output = pipeline_function(user_input)

            print("Evaluating response...")
            # Evaluate the final response (returns scores, explanation, raw_text)
            scores, explanation, raw_eval, overall = evaluate_response(user_input, pipeline_output['final_response'])

            # Calculate average score ignoring zeros
            valid = [v for v in scores.values() if v and v > 0]
            avg_score = (sum(valid) / len(valid)) if valid else 0

            # Combine all data
            row = {
                "sample_id": i,
                "user_input": user_input,
                "transformed_text": pipeline_output['transformed_text'],
                "extracted_memory": pipeline_output['extracted_memory'],
                "professional_response": pipeline_output['professional_response'],
                "final_response": pipeline_output['final_response'],
                "raw_eval_response": raw_eval,
                "eval_explanation": explanation,
                "overall_score": overall,
                **scores,
                "average_score": round(avg_score, 2)
            }

            results.append(row)

            print(f"Average Score (metrics mean): {avg_score:.2f}/10")
            print(f"Scores: {scores}")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_csv}")
    print(f"Total samples evaluated: {len(results)}")

    if results:
        overall_avg = sum(r['average_score'] for r in results) / len(results)
        print(f"Overall Average Score: {overall_avg:.2f}/5")

        # Print individual metric averages
        print("\nAverage scores by metric:")
        for metric in ["active_listening", "empathy_validation", "safety_trustworthiness",
                       "openmindedness_nonjudgment", "clarity_encouragement",
                       "boundaries_ethical", "holistic_approach"]:
            metric_avg = sum(r[metric] for r in results) / len(results)
            print(f"  {metric}: {metric_avg:.2f}/5")

    print(f"{'='*60}")

    return output_csv

# Example usage
if __name__ == "__main__":
    from pipeline import mental_health_pipeline_detailed

    # Run evaluation on 5 samples
    evaluate_pipeline(mental_health_pipeline_detailed, num_samples=5)
