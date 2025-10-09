from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import time
from openai import RateLimitError

# Load environment variables from .env file
load_dotenv()

# Load the dataset
ds = load_dataset("ShenLab/MentalChat16K")

# Initialize the LLM (API key loaded from .env file)
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Memory file path
MEMORY_FILE = Path("memory.txt")

def ensure_memory_file():
    """Ensure memory.txt exists"""
    if not MEMORY_FILE.exists():
        MEMORY_FILE.touch()

def read_memory():
    """Read existing memory from memory.txt"""
    ensure_memory_file()
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return f.read()

def append_memory(new_memory):
    """Append new memory to memory.txt"""
    ensure_memory_file()
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(new_memory + "\n")

def clear_memory():
    """Clear all memory from memory.txt (use when starting with a new person)"""
    ensure_memory_file()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write("")

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
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt

            # Extract wait time from error message or use exponential backoff
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            print(f"Rate limit hit. Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
        except Exception as e:
            # Re-raise non-rate-limit errors immediately
            raise

def llm1_transform_and_extract(user_input):
    """
    LLM 1: Transform user input to third-person and extract important facts
    Returns: (transformed_text, extracted_memory)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a clinical assistant that transforms patient statements into third-person clinical notes.

Your tasks:
1. Transform everything the person said into third person, referring to them as "the patient"
2. Extract and identify important facts about the patient that should be remembered
   - Include SPECIFIC FACTS: names, dates, events, relationships, medications, diagnoses, etc.
   - Include BROADER CONTEXT: patterns, emotional states, cognitive observations, behavioral trends
3. Pay special attention to cognitive elements and note anything unusual about their cognition, thought patterns, or mental state

Format your response as:
TRANSFORMED TEXT:
[The third-person transformation here]

MEMORY TO STORE:
SPECIFIC FACTS:
- [List specific, concrete facts like names, dates, events, medications, etc.]

BROADER CONTEXT:
- [List patterns, emotional states, cognitive observations, behavioral trends]"""),
        ("human", "{input}")
    ])

    chain = prompt | llm
    response = invoke_with_retry(chain, {"input": user_input})

    # Parse the response
    content = response.content
    transformed_section = content.split("MEMORY TO STORE:")[0].replace("TRANSFORMED TEXT:", "").strip()
    memory_section = content.split("MEMORY TO STORE:")[1].strip() if "MEMORY TO STORE:" in content else ""

    # Store the memory
    if memory_section:
        append_memory(memory_section)

    return transformed_section, memory_section

def llm2_generate_professional_response(transformed_text):
    """
    LLM 2: Generate professional, factual response using transformed text and memory
    """
    existing_memory = read_memory()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional mental health clinician. You provide factual, reasonable, and professional responses.

You have access to patient history and important facts that have been documented. Use this information along with the current clinical note to generate an appropriate, professional response.

Your response should be:
- Professional and clinical in tone
- Factual and evidence-based
- Reasonable and measured
- Considerate of the patient's wellbeing
- Informed by both current observation and historical context

Patient History and Important Facts:
{memory}"""),
        ("human", """Current Clinical Note:
{transformed_text}

Please provide a professional clinical response.""")
    ])

    chain = prompt | llm
    response = invoke_with_retry(chain, {
        "transformed_text": transformed_text,
        "memory": existing_memory if existing_memory else "No previous history available."
    })

    return response.content

def llm3_warmify_response(professional_response):
    """
    LLM 3: Transform professional response to warm and compassionate while retaining facts
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at transforming clinical language into warm, compassionate, direct communication while preserving all factual content.

Your task:
- Transform the professional clinical response into a direct, conversational response
- DO NOT write like a letter or formal message (avoid "Dear...", formal greetings, closings)
- Be direct, warm, and professional - like a therapist speaking naturally to someone
- Retain ALL factual elements and recommendations
- Make the tone empathetic, supportive, and human
- Use accessible language that makes the person feel heard and cared for
- Keep it concise and conversational
- Do not add new clinical content, only adjust the tone and warmth"""),
        ("human", """Professional Response:
{professional_response}

Transform this into a warm, direct, conversational response while keeping all the facts.""")
    ])

    chain = prompt | llm
    response = invoke_with_retry(chain, {"professional_response": professional_response})

    return response.content

def mental_health_pipeline(user_input, verbose=True):
    """
    Main pipeline function that processes user input through three LLMs:
    1. Transform to third-person and extract memory
    2. Generate professional response
    3. Transform to warm and compassionate tone

    Args:
        user_input (str): The user's input text
        verbose (bool): Whether to print progress information

    Returns:
        str: The final warm and compassionate response
    """
    if verbose:
        print("=" * 60)
        print("MENTAL HEALTH SUPPORT PIPELINE")
        print("=" * 60)

    # Step 1: Transform and extract memory
    if verbose:
        print("\n[Step 1] Transforming input to clinical note and extracting memory...")
    transformed_text, extracted_memory = llm1_transform_and_extract(user_input)
    if verbose:
        print(f"Transformed Text: {transformed_text[:100]}...")
        if extracted_memory:
            print(f"Memory Extracted: {extracted_memory[:100]}...")

    # Step 2: Generate professional response
    if verbose:
        print("\n[Step 2] Generating professional clinical response...")
    professional_response = llm2_generate_professional_response(transformed_text)
    if verbose:
        print(f"Professional Response: {professional_response[:100]}...")

    # Step 3: Transform to warm and compassionate
    if verbose:
        print("\n[Step 3] Transforming to warm and compassionate tone...")
    final_response = llm3_warmify_response(professional_response)

    if verbose:
        print("\n" + "=" * 60)
        print("FINAL RESPONSE:")
        print("=" * 60)
        print(final_response)
        print("=" * 60)

    return final_response

def mental_health_pipeline_detailed(user_input):
    """
    Pipeline that returns detailed intermediate outputs for evaluation.

    Args:
        user_input (str): The user's input text

    Returns:
        dict: Contains all intermediate outputs and final response
    """
    # Step 1: Transform and extract memory
    transformed_text, extracted_memory = llm1_transform_and_extract(user_input)

    # Step 2: Generate professional response
    professional_response = llm2_generate_professional_response(transformed_text)

    # Step 3: Transform to warm and compassionate
    final_response = llm3_warmify_response(professional_response)

    return {
        "user_input": user_input,
        "transformed_text": transformed_text,
        "extracted_memory": extracted_memory,
        "professional_response": professional_response,
        "final_response": final_response
    }

# Example usage
if __name__ == "__main__":
    # Example user input
    example_input = """I've been feeling really anxious lately. I can't seem to focus on anything
    and I keep forgetting things. Yesterday I forgot my meeting and today I left my keys in the door.
    I'm worried something is wrong with my memory."""

    # Run the pipeline
    response = mental_health_pipeline(example_input)
