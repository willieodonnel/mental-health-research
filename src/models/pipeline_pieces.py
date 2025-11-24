"""
Pipeline Components - Individual Reusable Pieces

This module provides individual pipeline components that can be combined
in different ways for ablation testing. Each component uses the exact same
prompts as the original pipeline.
"""

from typing import Dict, Any


def create_clinical_description_prompt(user_input: str) -> str:
    """
    Create the prompt for converting user input to clinical description.

    Args:
        user_input: The user's mental health concern in first person

    Returns:
        Prompt string for clinical description generation
    """
    return f"""Convert this to third-person clinical language:
"{user_input}"

Change "I" to "The patient", keep it concise and clinical."""


def create_professional_opinion_prompt(input_text: str) -> str:
    """
    Create the prompt for generating professional opinion.

    Args:
        input_text: Either the clinical description or raw user input

    Returns:
        Prompt string for professional opinion generation
    """
    return f"""As a mental health professional, provide a brief assessment of:
{input_text}

Identify key concerns and provide professional opinion."""


def create_final_response_prompt(user_input: str, context: str, context_type: str = "professional") -> str:
    """
    Create the prompt for generating the final empathetic response.

    Args:
        user_input: Original user concern
        context: Either professional opinion or clinical description
        context_type: Either "professional" or "clinical" to determine label

    Returns:
        Prompt string for final response generation
    """
    context_label = "Professional context" if context_type == "professional" else "Clinical context"

    return f"""You are an empathetic counselor. Using the {context_type} context below, respond helpfully to the patient's concern.

Original concern: {user_input}

{context_label}: {context}

Provide a compassionate, helpful response:"""


def run_clinical_description(model, tokenizer, generate_fn, user_input: str) -> str:
    """
    Component 1: Convert user input to clinical description.

    Args:
        model: The loaded language model
        tokenizer: The model's tokenizer
        generate_fn: The generation function to use
        user_input: User's mental health concern

    Returns:
        Clinical description in third-person
    """
    prompt = create_clinical_description_prompt(user_input)
    return generate_fn(model, tokenizer, prompt)


def run_professional_opinion(model, tokenizer, generate_fn, input_text: str) -> str:
    """
    Component 2: Generate professional opinion from input.

    Args:
        model: The loaded language model
        tokenizer: The model's tokenizer
        generate_fn: The generation function to use
        input_text: Either clinical description or raw user input

    Returns:
        Professional opinion/assessment
    """
    prompt = create_professional_opinion_prompt(input_text)
    return generate_fn(model, tokenizer, prompt)


def run_final_response(model, tokenizer, generate_fn, user_input: str, context: str, context_type: str = "professional") -> str:
    """
    Component 3: Generate final empathetic response.

    Args:
        model: The loaded language model
        tokenizer: The model's tokenizer
        generate_fn: The generation function to use
        user_input: Original user concern
        context: Either professional opinion or clinical description
        context_type: Either "professional" or "clinical"

    Returns:
        Final empathetic response
    """
    prompt = create_final_response_prompt(user_input, context, context_type)
    return generate_fn(model, tokenizer, prompt)


# Export all public functions
__all__ = [
    'create_clinical_description_prompt',
    'create_professional_opinion_prompt',
    'create_final_response_prompt',
    'run_clinical_description',
    'run_professional_opinion',
    'run_final_response'
]
