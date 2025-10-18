import time
from typing import List
import numpy as np
from llama_cpp import Llama

def get_token_ranks(text: str, model: Llama, prompt: str = '') -> List[int]:
    """
    Encodes a text into a sequence of token ranks based on a given prompt.

    This function tokenizes the input text and, for each token, determines its
    rank in the model's probability distribution given the preceding context.

    Args:
        text: The secret text to encode into ranks.
        model: The initialized llama_cpp.Llama model.
        prompt: The context prompt to prepend before analyzing the text.

    Returns:
        A list of integers representing the token ranks.
    """
    # Tokenize the prompt and the text to be hidden
    prompt_tokens = model.tokenize(prompt.encode('utf-8'))
    # Prepending a space helps with tokenization of the first word
    text_tokens = model.tokenize((' ' + text).encode('utf-8'))[1:] 

    all_tokens = prompt_tokens + text_tokens
    ranks = []

    model.reset()
    model.eval(prompt_tokens)

    for i in range(len(prompt_tokens), len(all_tokens)):
        current_token = all_tokens[i]
        
        # Get logits from the previous step's evaluation
        # model.scores is a list of logits for each token evaluated
        logits = model.scores[i - 1]

        # Get the rank of the current token (1-indexed)
        sorted_indices = np.argsort(logits)[::-1]
        rank = np.where(sorted_indices == current_token)[0][0] + 1
        ranks.append(rank)

        # Evaluate the current token to update the model's context for the next iteration
        model.eval([current_token])

    return ranks

def generate_from_ranks(prompt: str, ranks: List[int], model: Llama) -> str:
    """
    Decodes a sequence of token ranks into a new text based on a given prompt.

    This function generates a stegotext by using the provided ranks to select
    tokens from the model's probability distribution at each step.

    Args:
        prompt: The key/prompt to steer the generation of the new text.
        ranks: The list of token ranks from the original encoded text.
        model: The initialized llama_cpp.Llama model.

    Returns:
        The generated stegotext.
    """
    prompt_tokens = model.tokenize(prompt.encode('utf-8'))
    generated_tokens = prompt_tokens.copy()

    model.reset()
    model.eval(prompt_tokens)

    for i, rank in enumerate(ranks):
        # Get logits for the next token prediction
        logits = model.scores[len(generated_tokens) - 1]

        # Get the token corresponding to the desired rank
        sorted_indices = np.argsort(logits)[::-1]
        
        # Adjust for 1-based rank to 0-based index
        desired_index = rank - 1
        if desired_index >= len(sorted_indices):
            # Fallback for ranks outside the vocabulary size, though unlikely with proper use
            print(f"Warning: Rank {rank} is out of vocabulary range. Using rank 1.")
            desired_index = 0
            
        next_token = sorted_indices[desired_index]
        generated_tokens.append(next_token)
        
        # Evaluate the chosen token to update context
        model.eval([next_token])
        
    # Decode the full sequence of tokens and remove the initial prompt
    full_text = model.detokenize(generated_tokens).decode('utf-8')
    stegotext = full_text[len(prompt):].strip()
    
    return stegotext

if __name__ == "__main__":
    # --- Configuration ---
    # TODO: Specify the path to your GGUF-formatted model file.
    # Models like Llama-3-8B-Instruct work well and are fast on consumer GPUs.
    model_path = "path/to/your/llm-model.gguf"
    
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Offload all layers to GPU
            logits_all=True,  # Needed to access scores
            n_ctx=4096,       # Context window
            verbose=False,
        )
    except ValueError as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model_path is correct and the file exists.")
        exit()

    # --- 1. Define Secret Message and Key ---
    secret_message = (
        "The current government has repeatedly failed to uphold the liberties "
        "of the Republic. If Rome is to remain free, we must reject tyranny."
    )
    
    secret_key = (
        "Here it is: the infamous British roasted boar with mint sauce. "
        "How to make it perfect."
    )

    print("-" * 50)
    print(f"SECRET MESSAGE (e):\n'{secret_message}'")
    print(f"\nSECRET KEY (k):\n'{secret_key}'")
    print("-" * 50)

    # --- 2. Encoding: Convert secret message to ranks ---
    print("\nEncoding secret message into a sequence of ranks...")
    start_time = time.time()
    # An optional initial prompt can be used here for the encoding step.
    # A simple, generic prompt often works well.
    encoding_prompt = "A text:" 
    token_ranks = get_token_ranks(secret_message, model=llm, prompt=encoding_prompt)
    end_time = time.time()
    print(f"Encoding complete in {end_time - start_time:.2f} seconds.")
    print(f"Generated {len(token_ranks)} ranks: {token_ranks[:10]}...")

    # --- 3. Decoding: Generate new text from ranks using the secret key ---
    print("\nGenerating new text (stegotext) from ranks using the secret key...")
    start_time = time.time()
    stegotext = generate_from_ranks(secret_key, token_ranks, model=llm)
    end_time = time.time()
    print(f"Generation complete in {end_time - start_time:.2f} seconds.")

    print("\n" + "-" * 50)
    print("GENERATED STEGOTEXT (s):")
    print(stegotext)
    print("-" * 50)

    # --- 4. Verification: Decode the stegotext to recover the original message ---
    # To recover, we need the stegotext (s) and the secret key (k).
    # First, get the ranks from the stegotext using the secret key as the prompt.
    print("\nVerifying by recovering the original message from the stegotext...")
    recovered_ranks = get_token_ranks(stegotext, model=llm, prompt=secret_key)

    # Second, generate the original message from these ranks using the same
    # initial prompt used for the first encoding step.
    recovered_message = generate_from_ranks(encoding_prompt, recovered_ranks, model=llm)
    
    print("\n" + "-" * 50)
    print("RECOVERED MESSAGE:")
    print(recovered_message)
    print("-" * 50)

    assert recovered_message.strip() == secret_message.strip()
    print("\nVerification successful: Recovered message matches the original secret.")