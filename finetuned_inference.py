'''
Here we host a version of the finetuned version of Llama-3.2-1B-Instruct from huggingface
'''

from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import torch


def load_model():
    model_name = "khazarai/MentalChat-16K"
    base_model_name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, 
                                      dtype=torch.float16,
                                      device_map='auto')

    model = PeftModel.from_pretrained(base_model, model_name)

    print('Loaded finetuned Llama-3.2-1B-Instruct model!')
    return model, tokenizer


def generate(model, tokenizer, question):
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    system = """You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description.
    The assistant gives helpful, comprehensive, and appropriate answers to the user's questions.
    """

    messages = [
    {"role" : "system", "content" : system},
    {"role" : "user", "content" : question}
    ]

    response = pipe(messages)

    return response


def run_pipeline(prompt: str):
    model, tokenizer = load_model()

    response = generate(model, tokenizer, prompt)

if __name__ == "__main__":
    run_pipeline("I've been really scared that my hands aren't clean. I can't stop washing them because I'm scared they'll get dirty and I'll get sick and die.")
