"""
Ablation Pipeline - Extremely Simple 3-Component Implementation

Components:
1. Convert user input to third-person clinical language
2. Generate professional opinion from clinical description
3. Generate final response using both contexts
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model():
    """Load Mistral model once for all components."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print("Loading model for pipeline...")

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
    """Generate response - same as main.py."""
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


def run_pipeline(user_input):
    """
    Run the 3-component pipeline on user input.

    Args:
        user_input: Mental health question from user

    Returns:
        Dictionary with all components and final response
    """
    # Load model once
    model, tokenizer = load_model()

    print("\n" + "="*60)
    print("Running 3-Component Pipeline")
    print("="*60)

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

    # Component 3: Final response
    response_prompt = f"""You are an empathetic counselor. Using the professional context below, respond helpfully to the patient's concern.

Original concern: {user_input}

Professional context: {professional_opinion}

Provide a compassionate, helpful response:"""

    final_response = generate(model, tokenizer, response_prompt)
    print("\n3. Final Response:")
    print(final_response)

    return {
        "user_input": user_input,
        "clinical_description": clinical_description,
        "professional_opinion": professional_opinion,
        "final_response": final_response
    }


def main():
    """Simple test of the pipeline."""

    # Test input
    test_input = "I've been really scared that my hands aren't clean. I can't stop washing them because I'm scared they'll get dirty and I'll get sick and die. They started to peel and are red. I'm fine in that case, right?"

    # Run pipeline
    result = run_pipeline(test_input)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    # Interactive mode
    print("\nEnter 'quit' to exit, or type a message:")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        result = run_pipeline(user_input)


if __name__ == "__main__":
    main()