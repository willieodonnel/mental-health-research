"""
Mistral-7B Local Inference Pipeline

This is an extremely basic pipeline for exploring Mistral-7B-Instruct locally.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    """Load Mistral-7B-Instruct model and tokenizer."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print("Loading Mistral-7B-Instruct...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model (will auto-detect GPU)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,  # Use FP16 for RTX 5080
        device_map="auto"            # Automatically use GPU if available
    )

    print(f"Model loaded successfully! Using {model.device}")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_length=1024):
    """Generate a response from the model."""
    # Format prompt in Mistral instruction format
    formatted_prompt = f"[INST] {prompt} [/INST]"

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.5,
            do_sample=True,
            top_p=0.95
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response (after [/INST])
    response = response.split("[/INST]")[-1].strip()

    return response


def main():
    """Basic demo - load model and test with one question."""
    # Load the model
    model, tokenizer = load_model()

    # Test prompt
    prompt = "I can't stop washing my hands because I feel unclean. If I stop, my hands might get dirty. Last week, my hands started to peel and were red. There's nothing wrong with this, right?"

    print("\n" + "="*60)
    print("PROMPT:", prompt)
    print("="*60)

    # Generate response
    response = generate(model, tokenizer, prompt)

    print("\nRESPONSE:")
    print("-"*60)
    print(response)
    print("-"*60)

    # Interactive mode (optional)
    print("\nEnter 'quit' to exit, or type a message to chat:")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        response = generate(model, tokenizer, user_input)
        print(f"\nMistral: {response}")


if __name__ == "__main__":
    main()