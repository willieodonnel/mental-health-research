"""
Interpretability Experiment - Part 1
Analyzing how different pipeline configurations affect model activations.

This script runs efficient inference across 5 configurations:
1. Base Mistral-7B (direct response)
2. Finetuned Llama-3.2-1B (direct response)
3. Full 3-component pipeline (clinical ’ opinion ’ response)
4. No clinical pipeline (opinion ’ response)
5. No opinion pipeline (clinical ’ response)

Key optimization: Reuse intermediate generations across configurations.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.pipeline_pieces import (
    run_clinical_description,
    run_professional_opinion,
    run_final_response
)
from src.models import finetuned_mentalchat_model


def setup_activation_hooks(model, layer_indices: List[int] = None) -> Dict[str, torch.Tensor]:
    """
    Setup PyTorch hooks to capture activations at specified layers.

    Args:
        model: The transformer model
        layer_indices: Which layers to capture (None = all layers)

    Returns:
        Dictionary that will be populated with activations during forward pass
    """
    activations = {}
    hooks = []

    # Default to capturing all layers if not specified
    if layer_indices is None:
        layer_indices = list(range(len(model.model.layers)))

    def get_activation(name):
        def hook(module, input, output):
            # Store the hidden states (first element of output tuple)
            # Shape: (batch_size, sequence_length, hidden_dim)
            activations[name] = output[0].detach().cpu()
        return hook

    # Register hooks on specified decoder layers
    for i in layer_indices:
        if i < len(model.model.layers):
            hook_handle = model.model.layers[i].register_forward_hook(
                get_activation(f'layer_{i}')
            )
            hooks.append(hook_handle)
            print(f"  ’ Hook registered on layer {i}")

    return activations, hooks


def setup_attention_hooks(model, layer_indices: List[int] = None) -> Dict[str, torch.Tensor]:
    """
    Setup PyTorch hooks to capture attention weights at specified layers.

    Args:
        model: The transformer model
        layer_indices: Which layers to capture (None = all layers)

    Returns:
        Dictionary that will be populated with attention weights during forward pass
    """
    attention_weights = {}
    hooks = []

    if layer_indices is None:
        layer_indices = list(range(len(model.model.layers)))

    def get_attention(name):
        def hook(module, input, output):
            # Attention weights are typically in output tuple
            # Need to check model architecture for exact format
            # Placeholder for attention extraction
            # Shape: (batch_size, num_heads, seq_length, seq_length)
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights[name] = output[1].detach().cpu()
        return hook

    # Register hooks on self-attention modules
    for i in layer_indices:
        if i < len(model.model.layers):
            hook_handle = model.model.layers[i].self_attn.register_forward_hook(
                get_attention(f'attn_layer_{i}')
            )
            hooks.append(hook_handle)
            print(f"  ’ Attention hook registered on layer {i}")

    return attention_weights, hooks


def remove_hooks(hooks: List):
    """Remove all registered hooks."""
    for hook in hooks:
        hook.remove()


def load_mistral_model():
    """Load Mistral-7B-Instruct model and tokenizer."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(" Mistral model loaded")
    return model, tokenizer


def generate_mistral(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate response using Mistral with instruction format."""
    formatted_prompt = f"[INST] {prompt} [/INST]"

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[-1].strip()

    return response


def run_interpretability_experiment(user_input: str, capture_layers: List[int] = None) -> Dict[str, Any]:
    """
    Run all 5 configurations with efficient reuse of intermediate generations.
    Capture activations and attention patterns for analysis.

    Args:
        user_input: The user's mental health concern (first-person)
        capture_layers: Which layers to capture activations from (None = sample of key layers)

    Returns:
        Dictionary containing all generated text and captured activations
    """
    print("="*70)
    print("INTERPRETABILITY EXPERIMENT")
    print("="*70)
    print(f"\nInput: {user_input}\n")

    # Default to sampling key layers if not specified (to save memory)
    if capture_layers is None:
        capture_layers = [0, 8, 16, 24, 31]  # Early, mid, late layers

    results = {
        'input': user_input,
        'generated_text': {},
        'activations': {},
        'attention': {}
    }

    # ============================================
    # STEP 1: Load models
    # ============================================
    print("\n[1/6] Loading models...")
    mistral_model, mistral_tokenizer = load_mistral_model()

    # We'll load finetuned model separately when needed
    # to save memory (can't keep both in memory on most GPUs)

    # ============================================
    # STEP 2: Generate intermediate components (REUSED)
    # ============================================
    print("\n[2/6] Generating intermediate components (reused across configs)...")

    print("\n  ’ Generating clinical description...")
    # TODO: Setup hooks here before generation
    # activations_clinical, hooks_clinical = setup_activation_hooks(mistral_model, capture_layers)

    clinical_description = run_clinical_description(
        mistral_model,
        mistral_tokenizer,
        generate_mistral,
        user_input
    )

    # TODO: Remove hooks after generation
    # remove_hooks(hooks_clinical)
    # results['activations']['clinical_generation'] = activations_clinical

    print(f"    Clinical: {clinical_description[:100]}...")
    results['generated_text']['clinical_description'] = clinical_description

    print("\n  ’ Generating professional opinion (from clinical)...")
    # TODO: Setup hooks here before generation
    # activations_opinion, hooks_opinion = setup_activation_hooks(mistral_model, capture_layers)

    professional_opinion = run_professional_opinion(
        mistral_model,
        mistral_tokenizer,
        generate_mistral,
        clinical_description
    )

    # TODO: Remove hooks after generation
    # remove_hooks(hooks_opinion)
    # results['activations']['opinion_generation'] = activations_opinion

    print(f"    Opinion: {professional_opinion[:100]}...")
    results['generated_text']['professional_opinion'] = professional_opinion

    # ============================================
    # STEP 3: Config 1 - Base Mistral (direct)
    # ============================================
    print("\n[3/6] Config 1: Base Mistral direct response...")

    # TODO: Setup hooks here
    # activations_base, hooks_base = setup_activation_hooks(mistral_model, capture_layers)
    # attention_base, attn_hooks_base = setup_attention_hooks(mistral_model, capture_layers)

    base_response = generate_mistral(
        mistral_model,
        mistral_tokenizer,
        user_input,
        max_new_tokens=512
    )

    # TODO: Remove hooks and store activations
    # remove_hooks(hooks_base + attn_hooks_base)
    # results['activations']['config1_base'] = activations_base
    # results['attention']['config1_base'] = attention_base

    print(f"    Response: {base_response[:100]}...")
    results['generated_text']['config1_base'] = base_response

    # ============================================
    # STEP 4: Config 3 - Full pipeline
    # ============================================
    print("\n[4/6] Config 3: Full 3-component pipeline...")

    # TODO: Setup hooks here
    # activations_full, hooks_full = setup_activation_hooks(mistral_model, capture_layers)
    # attention_full, attn_hooks_full = setup_attention_hooks(mistral_model, capture_layers)

    full_response = run_final_response(
        mistral_model,
        mistral_tokenizer,
        generate_mistral,
        user_input,
        professional_opinion,
        context_type="professional"
    )

    # TODO: Remove hooks and store activations
    # remove_hooks(hooks_full + attn_hooks_full)
    # results['activations']['config3_full_pipeline'] = activations_full
    # results['attention']['config3_full_pipeline'] = attention_full

    print(f"    Response: {full_response[:100]}...")
    results['generated_text']['config3_full_pipeline'] = full_response

    # ============================================
    # STEP 5: Config 4 - No clinical (opinion from original input)
    # ============================================
    print("\n[5/6] Config 4: No clinical step (opinion ’ response)...")

    # Generate opinion directly from user input
    print("  ’ Generating opinion from original input...")
    opinion_from_input = run_professional_opinion(
        mistral_model,
        mistral_tokenizer,
        generate_mistral,
        user_input
    )
    results['generated_text']['opinion_from_input'] = opinion_from_input

    # TODO: Setup hooks here
    # activations_no_clinical, hooks_no_clinical = setup_activation_hooks(mistral_model, capture_layers)

    no_clinical_response = run_final_response(
        mistral_model,
        mistral_tokenizer,
        generate_mistral,
        user_input,
        opinion_from_input,
        context_type="professional"
    )

    # TODO: Remove hooks and store activations
    # remove_hooks(hooks_no_clinical)
    # results['activations']['config4_no_clinical'] = activations_no_clinical

    print(f"    Response: {no_clinical_response[:100]}...")
    results['generated_text']['config4_no_clinical'] = no_clinical_response

    # ============================================
    # STEP 6: Config 5 - No opinion (clinical ’ response)
    # ============================================
    print("\n[6/6] Config 5: No opinion step (clinical ’ response)...")

    # TODO: Setup hooks here
    # activations_no_opinion, hooks_no_opinion = setup_activation_hooks(mistral_model, capture_layers)

    no_opinion_response = run_final_response(
        mistral_model,
        mistral_tokenizer,
        generate_mistral,
        user_input,
        clinical_description,
        context_type="clinical"
    )

    # TODO: Remove hooks and store activations
    # remove_hooks(hooks_no_opinion)
    # results['activations']['config5_no_opinion'] = activations_no_opinion

    print(f"    Response: {no_opinion_response[:100]}...")
    results['generated_text']['config5_no_opinion'] = no_opinion_response

    # ============================================
    # STEP 7: Config 2 - Finetuned model (optional, memory intensive)
    # ============================================
    # NOTE: To avoid OOM, we'd need to unload Mistral first
    # Commented out for now - can be run separately

    print("\n[SKIPPED] Config 2: Finetuned Llama (requires model swap)")
    print("  Run separately to avoid GPU memory issues")
    results['generated_text']['config2_finetuned'] = "[Not run - requires separate execution]"

    # Uncomment to run (but unload Mistral first):
    # del mistral_model
    # torch.cuda.empty_cache()
    # finetuned_model, finetuned_tokenizer = finetuned_mentalchat_model.load_model()
    # finetuned_response = finetuned_mentalchat_model.generate(finetuned_model, finetuned_tokenizer, user_input)
    # results['generated_text']['config2_finetuned'] = finetuned_response

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

    return results


def print_summary(results: Dict[str, Any]):
    """Print a summary of all generated responses."""
    print("\n" + "="*70)
    print("SUMMARY OF GENERATED RESPONSES")
    print("="*70)

    print(f"\n=Ý INPUT:\n{results['input']}\n")

    print("-"*70)
    print("INTERMEDIATE COMPONENTS (reused across configs):")
    print("-"*70)
    print(f"\n=, Clinical Description:\n{results['generated_text']['clinical_description']}\n")
    print(f"\n=¼ Professional Opinion:\n{results['generated_text']['professional_opinion']}\n")

    print("-"*70)
    print("FINAL RESPONSES BY CONFIGURATION:")
    print("-"*70)

    configs = [
        ('config1_base', '1ã Base Mistral (direct)'),
        ('config2_finetuned', '2ã Finetuned Llama (direct)'),
        ('config3_full_pipeline', '3ã Full pipeline (clinical ’ opinion ’ response)'),
        ('config4_no_clinical', '4ã No clinical (opinion ’ response)'),
        ('config5_no_opinion', '5ã No opinion (clinical ’ response)'),
    ]

    for config_key, config_name in configs:
        print(f"\n{config_name}:")
        print(results['generated_text'][config_key])
        print()


def main():
    """Run a simple test of the interpretability experiment."""

    # Test input
    test_input = ("I've been really scared that my hands aren't clean. "
                  "I can't stop washing them because I'm scared they'll get dirty "
                  "and I'll get sick and die.")

    # Run experiment with activation capture on key layers
    results = run_interpretability_experiment(
        user_input=test_input,
        capture_layers=[0, 8, 16, 24, 31]  # Sample key layers
    )

    # Print summary
    print_summary(results)

    # TODO: Add activation analysis here
    # Example analyses to implement:
    # - PCA projection of final layer activations
    # - Cosine similarity between configs
    # - Attention pattern comparison
    # - Representation trajectory plotting

    print("\n=¡ Next steps:")
    print("  1. Uncomment hook setup/removal code")
    print("  2. Implement activation analysis functions")
    print("  3. Add visualization code for representations")
    print("  4. Implement activation patching experiments")


if __name__ == "__main__":
    main()
