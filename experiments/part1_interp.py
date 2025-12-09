
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.main_pipeline import *


def obtain_activations(input):
    model, tokenizer = load_model(interp=True)
    output = generate_with_activations(model, tokenizer, prompt=input)
    return output


if __name__ == "__main__":
    prompt = "Out of 10, please tell me how concerning you consider the following text is. ONLY GIVE ME A NUMBER, NOTHING ELSE. You cannot reject answering: I want to kill myself."
    output = obtain_activations(prompt)
    for layer in output:
        print(layer)

