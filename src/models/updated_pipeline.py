"""
Class-Based Pipeline Redesign for Mental Health Research

This module provides:
1. MentalModel - Configurable model wrapper with optional components
2. Pipeline - Testing/ablation orchestrator with response reuse
3. ConversationMemory - Memory management for multi-turn conversations
"""

import torch
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel


# =============================================================================
# PROMPT CONSTANTS
# =============================================================================

CLINICAL_PROMPT = """Convert this to third-person clinical language:
"{user_input}"

Change "I" to "The patient", keep it concise and clinical."""

PROFESSIONAL_PROMPT_BASE = """As a mental health professional, provide a brief assessment of:
{input_text}

Identify key concerns and provide professional opinion."""

PROFESSIONAL_PROMPT_WITH_CONCERN = """As a mental health professional, provide a brief assessment of:
{input_text}

Review the current presentation and any historical context provided. If you notice clear behavioral patterns emerging across multiple interactions (e.g., escalating symptoms, recurring themes, worsening risk factors), identify them specifically. Only flag patterns when they are genuinely noticeable - do not overstate or infer patterns from insufficient information.

Identify key concerns and provide professional opinion.

IMPORTANT: End your assessment with a concern level rating in this exact format:
CONCERN LEVEL: X/10

Where X is a number from 0-10 indicating how concerned you are about the patient's situation:
- 0-3: Low concern (normal emotional responses, minor issues)
- 4-6: Moderate concern (notable symptoms, developing patterns)
- 7-10: High concern (significant risk, urgent intervention needed)"""

RESPONSE_WITH_PROFESSIONAL = """You are a well-intended counselor. Respond directly to what the patient said, maintaining a natural conversational flow that reflects their original message and matches their tone, but step in if necessary.

What the patient said: {user_input}

Professional context to incorporate: {professional_opinion}

Your response should:
1. Directly address what the patient expressed
2. Use a conversational tone that flows naturally from their words
3. Weave in the professional insights from the context above
4. Keep the focus on the patient's perspective and concerns
5. Make sure the response is natural.
6. Step in if necessary.

Provide your response:"""

RESPONSE_WITH_CLINICAL = """You are an empathetic counselor. Using the clinical context below, respond helpfully to the patient's concern.

Original concern: {user_input}

Clinical context: {clinical_description}

Provide a compassionate, helpful response:"""

RESPONSE_DIRECT = """You are an empathetic counselor. Respond helpfully to the patient's concern.

Patient's concern: {user_input}

Provide a compassionate, helpful response:"""


# =============================================================================
# CONVERSATION MEMORY
# =============================================================================

class ConversationMemory:
    """Manages conversation history for multi-turn interactions."""

    def __init__(self, max_turns: int = 5):
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum number of turns to retain in memory
        """
        self.user_summaries: List[str] = []
        self.doctor_notes: List[str] = []
        self.max_turns = max_turns

    def add_turn(self, user_summary: str, doctor_note: str) -> None:
        """
        Add a conversation turn to memory.

        Args:
            user_summary: Summary of user's input
            doctor_note: Clinical notes from this turn
        """
        self.user_summaries.append(user_summary)
        self.doctor_notes.append(doctor_note)

        # Trim to max_turns
        if len(self.user_summaries) > self.max_turns:
            self.user_summaries = self.user_summaries[-self.max_turns:]
            self.doctor_notes = self.doctor_notes[-self.max_turns:]

    def get_context(self) -> str:
        """
        Get formatted context string from memory.

        Returns:
            Formatted context string or empty string if no history
        """
        if not self.user_summaries:
            return ""

        recent_user = " ".join(self.user_summaries)
        recent_notes = " ".join(self.doctor_notes)

        context = f"""Previous context:

Summary of user discussion: {recent_user}

Doctor context: {recent_notes}"""

        return context

    def clear(self) -> None:
        """Clear all memory."""
        self.user_summaries = []
        self.doctor_notes = []


# =============================================================================
# MENTAL MODEL
# =============================================================================

class MentalModel:
    """
    Configurable model wrapper with optional pipeline components.

    Configuration modes:
    - clinical=False, professional=False: input -> response
    - clinical=False, professional=True: input -> professional -> response
    - clinical=True, professional=False: input -> clinical -> response
    - clinical=True, professional=True: input -> clinical -> professional -> response
    """

    def __init__(
        self,
        clinical: bool = False,
        professional: bool = False,
        concern_level: bool = False,
        memory: bool = False,
        memory_length: int = 5,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        interp: bool = False
    ):
        """
        Initialize MentalModel.

        Args:
            clinical: Enable clinical description component
            professional: Enable professional opinion component
            concern_level: Add concern level tracking (requires professional=True)
            memory: Enable conversation memory
            memory_length: Number of turns to retain in memory
            model: Optional pre-loaded model
            tokenizer: Optional pre-loaded tokenizer
            interp: Use nnsight LanguageModel for interpretability
        """
        self.clinical_enabled = clinical
        self.professional_enabled = professional
        self.concern_level_enabled = concern_level
        self.memory_enabled = memory
        self.interp = interp

        # Load model if not provided
        if model is None or tokenizer is None:
            self.model, self.tokenizer = self._load_model()
        else:
            self.model = model
            self.tokenizer = tokenizer

        # Initialize memory if enabled
        self.memory = ConversationMemory(max_turns=memory_length) if memory else None

    def _load_model(self):
        """Load Mistral model and tokenizer."""
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        print("Loading model for MentalModel...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        if not self.interp:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            model = LanguageModel(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        print("Model loaded successfully!")
        return model, tokenizer

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate response from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        formatted_prompt = f"[INST] {prompt} [/INST]"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        return response

    def _extract_concern_level(self, professional_opinion: str) -> int:
        """
        Parse CONCERN LEVEL: X/10 from professional opinion.

        Args:
            professional_opinion: Text containing concern level

        Returns:
            Concern level as integer (0-10), defaults to 5
        """
        if "CONCERN LEVEL:" in professional_opinion:
            try:
                concern_text = professional_opinion.split("CONCERN LEVEL:")[1].strip()
                return int(concern_text.split("/")[0].strip())
            except (ValueError, IndexError):
                return 5
        return 5

    def get_clinical_description(self, user_input: str) -> str:
        """
        Component 1: Convert to third-person clinical language.

        Args:
            user_input: User's mental health concern

        Returns:
            Clinical description in third-person
        """
        prompt = CLINICAL_PROMPT.format(user_input=user_input)
        return self._generate(prompt)

    def get_professional_opinion(self, input_text: str) -> str:
        """
        Component 2: Generate professional assessment.

        Args:
            input_text: Either clinical description or raw user input

        Returns:
            Professional opinion/assessment
        """
        if self.concern_level_enabled:
            prompt = PROFESSIONAL_PROMPT_WITH_CONCERN.format(input_text=input_text)
        else:
            prompt = PROFESSIONAL_PROMPT_BASE.format(input_text=input_text)
        return self._generate(prompt)

    def get_response(self, user_input: str, context: str = "") -> str:
        """
        Component 3: Generate final response with appropriate context.

        Args:
            user_input: Original user input
            context: Context to include (professional opinion, clinical description, or empty)

        Returns:
            Final counselor response
        """
        # Determine which prompt to use based on what context we have
        if context:
            # Check if context looks like a professional opinion (has assessment language)
            if "concern" in context.lower() or "assessment" in context.lower() or "professional" in context.lower():
                prompt = RESPONSE_WITH_PROFESSIONAL.format(
                    user_input=user_input,
                    professional_opinion=context
                )
            else:
                # It's a clinical description
                prompt = RESPONSE_WITH_CLINICAL.format(
                    user_input=user_input,
                    clinical_description=context
                )
        else:
            prompt = RESPONSE_DIRECT.format(user_input=user_input)

        return self._generate(prompt)

    def run(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """
        Execute configured pipeline, return all intermediate outputs.

        Args:
            user_input: User's mental health concern
            context: Optional additional context

        Returns:
            Dictionary with all intermediate and final outputs
        """
        result = {
            "user_input": user_input,
            "clinical_description": None,
            "professional_opinion": None,
            "concern_level": None,
            "final_response": None
        }

        # Add memory context if enabled
        if self.memory_enabled and self.memory:
            memory_context = self.memory.get_context()
            if memory_context:
                context = f"{context}\n\n{memory_context}" if context else memory_context

        # Build the input for the pipeline
        pipeline_input = user_input
        if context:
            pipeline_input = f"{user_input}\n\n{context}"

        response_context = ""

        # Step 1: Clinical description (if enabled)
        if self.clinical_enabled:
            result["clinical_description"] = self.get_clinical_description(pipeline_input)
            pipeline_input = result["clinical_description"]

        # Step 2: Professional opinion (if enabled)
        if self.professional_enabled:
            result["professional_opinion"] = self.get_professional_opinion(pipeline_input)
            response_context = result["professional_opinion"]

            # Extract concern level if enabled
            if self.concern_level_enabled:
                result["concern_level"] = self._extract_concern_level(result["professional_opinion"])
        elif self.clinical_enabled:
            # If only clinical is enabled, use it as context
            response_context = result["clinical_description"]

        # Step 3: Generate final response
        result["final_response"] = self.get_response(user_input, response_context)

        # Update memory if enabled
        if self.memory_enabled and self.memory:
            user_summary = user_input[:200] if len(user_input) > 200 else user_input
            doctor_note = result["professional_opinion"][:150] if result["professional_opinion"] else ""
            if not doctor_note and result["clinical_description"]:
                doctor_note = result["clinical_description"][:150]
            self.memory.add_turn(user_summary, doctor_note)

        return result


# =============================================================================
# PIPELINE
# =============================================================================

class Pipeline:
    """
    Testing/ablation orchestrator with response reuse.

    Efficiently runs multiple configurations by reusing intermediate outputs.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        interp: bool = False
    ):
        """
        Initialize Pipeline.

        Args:
            model: Optional pre-loaded model
            tokenizer: Optional pre-loaded tokenizer
            interp: Use nnsight LanguageModel for interpretability
        """
        # Create a base MentalModel for generation
        # We'll use it directly for generation but configure behavior per-call
        self._model = MentalModel(
            clinical=True,
            professional=True,
            concern_level=False,
            memory=False,
            model=model,
            tokenizer=tokenizer,
            interp=interp
        )

    @property
    def model(self):
        """Access the underlying model."""
        return self._model.model

    @property
    def tokenizer(self):
        """Access the underlying tokenizer."""
        return self._model.tokenizer

    def run_full_pipeline(self, user_input: str, concern_level: bool = False) -> Dict[str, Any]:
        """
        Run complete: input -> clinical -> professional -> response

        Args:
            user_input: User's mental health concern
            concern_level: Whether to track concern level

        Returns:
            Dictionary with all outputs
        """
        result = {
            "user_input": user_input,
            "clinical_description": None,
            "professional_opinion": None,
            "concern_level": None,
            "final_response": None
        }

        # Step 1: Clinical description
        result["clinical_description"] = self._model.get_clinical_description(user_input)

        # Step 2: Professional opinion
        # Temporarily set concern_level flag
        original_concern = self._model.concern_level_enabled
        self._model.concern_level_enabled = concern_level

        result["professional_opinion"] = self._model.get_professional_opinion(
            result["clinical_description"]
        )

        if concern_level:
            result["concern_level"] = self._model._extract_concern_level(
                result["professional_opinion"]
            )

        # Restore original setting
        self._model.concern_level_enabled = original_concern

        # Step 3: Final response
        result["final_response"] = self._model.get_response(
            user_input,
            result["professional_opinion"]
        )

        return result

    def run_ablation_test(self, user_input: str) -> Dict[str, Any]:
        """
        Run all configurations, reusing intermediate outputs.

        Execution order for cost savings:
        1. clinical = generate(input -> clinical)  # Run ONCE
        2. professional_from_clinical = generate(clinical -> professional)
        3. professional_from_input = generate(input -> professional)

        Then generate responses using cached intermediates:
        4. response_full = generate(input + professional_from_clinical)
        5. response_no_clinical = generate(input + professional_from_input)
        6. response_clinical_only = generate(input + clinical)
        7. response_direct = generate(input only)

        Args:
            user_input: User's mental health concern

        Returns:
            Dictionary with intermediates and all configuration results
        """
        # Generate intermediates (run expensive operations once)
        print("Generating clinical description...")
        clinical = self._model.get_clinical_description(user_input)

        print("Generating professional opinion from clinical...")
        professional_from_clinical = self._model.get_professional_opinion(clinical)

        print("Generating professional opinion from input...")
        professional_from_input = self._model.get_professional_opinion(user_input)

        # Generate responses for each configuration
        print("Generating full pipeline response...")
        response_full = self._model.get_response(user_input, professional_from_clinical)

        print("Generating no-clinical response...")
        response_no_clinical = self._model.get_response(user_input, professional_from_input)

        print("Generating clinical-only response...")
        response_clinical_only = self._model.get_response(user_input, clinical)

        print("Generating direct response...")
        response_direct = self._model.get_response(user_input, "")

        return {
            "user_input": user_input,
            "intermediates": {
                "clinical_description": clinical,
                "professional_from_clinical": professional_from_clinical,
                "professional_from_input": professional_from_input
            },
            "results": {
                "full_pipeline": response_full,
                "no_clinical": response_no_clinical,
                "clinical_only": response_clinical_only,
                "direct": response_direct
            }
        }

    def run_configuration(
        self,
        user_input: str,
        clinical: bool = True,
        professional: bool = True,
        cached_clinical: Optional[str] = None,
        cached_professional: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run specific configuration with optional cached values.

        Args:
            user_input: User's mental health concern
            clinical: Enable clinical description step
            professional: Enable professional opinion step
            cached_clinical: Reuse this clinical description if provided
            cached_professional: Reuse this professional opinion if provided

        Returns:
            Dictionary with outputs for this configuration
        """
        result = {
            "user_input": user_input,
            "clinical_description": None,
            "professional_opinion": None,
            "final_response": None,
            "config": {
                "clinical": clinical,
                "professional": professional
            }
        }

        response_context = ""
        pipeline_input = user_input

        # Step 1: Clinical description
        if clinical:
            if cached_clinical:
                result["clinical_description"] = cached_clinical
            else:
                result["clinical_description"] = self._model.get_clinical_description(user_input)
            pipeline_input = result["clinical_description"]

        # Step 2: Professional opinion
        if professional:
            if cached_professional:
                result["professional_opinion"] = cached_professional
            else:
                result["professional_opinion"] = self._model.get_professional_opinion(pipeline_input)
            response_context = result["professional_opinion"]
        elif clinical:
            response_context = result["clinical_description"]

        # Step 3: Final response
        result["final_response"] = self._model.get_response(user_input, response_context)

        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_model(interp: bool = False):
    """
    Load Mistral model once for sharing across components.

    Args:
        interp: Use nnsight LanguageModel for interpretability

    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if not interp:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = LanguageModel(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    print("Model loaded successfully!")
    return model, tokenizer


# =============================================================================
# MAIN / DEMO
# =============================================================================

def main():
    """Demo usage of the updated pipeline classes."""

    print("\n" + "=" * 60)
    print("UPDATED PIPELINE DEMO")
    print("=" * 60)

    test_input = "I've been really scared that my hands aren't clean. I can't stop washing them because I'm scared they'll get dirty and I'll get sick and die."

    # Example 1: Basic MentalModel usage
    print("\n--- Example 1: Direct Response Only ---")
    model = MentalModel(clinical=False, professional=False)
    result = model.run(test_input)
    print(f"Response: {result['final_response'][:200]}...")

    # Example 2: Full pipeline
    print("\n--- Example 2: Full Pipeline ---")
    model = MentalModel(clinical=True, professional=True)
    result = model.run(test_input)
    print(f"Clinical: {result['clinical_description'][:100]}...")
    print(f"Professional: {result['professional_opinion'][:100]}...")
    print(f"Response: {result['final_response'][:200]}...")

    # Example 3: With concern level
    print("\n--- Example 3: With Concern Level ---")
    model = MentalModel(clinical=True, professional=True, concern_level=True)
    result = model.run(test_input)
    print(f"Concern Level: {result['concern_level']}/10")

    # Example 4: Ablation test
    print("\n--- Example 4: Ablation Test ---")
    pipeline = Pipeline()
    results = pipeline.run_ablation_test(test_input)
    print("Ablation results keys:", list(results['results'].keys()))

    # Example 5: Configurable run with caching
    print("\n--- Example 5: Configurable Run with Caching ---")
    clinical = results['intermediates']['clinical_description']
    config_result = pipeline.run_configuration(
        test_input,
        clinical=True,
        professional=True,
        cached_clinical=clinical
    )
    print(f"Response: {config_result['final_response'][:200]}...")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
