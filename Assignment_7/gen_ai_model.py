# gen_ai_model.py
# ----------------
# This script defines a simple generative AI model using Hugging Face GPT-2 and provides test cases for text generation.
# It can be used as a standalone script or imported as a module in other applications (e.g., FastAPI server).

# Import necessary libraries for the generative AI model
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Select device: use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)

class SimpleGenerativeAI:
    """
    A simple generative AI model using GPT-2 for text generation.
    Handles model loading, text generation, and performance monitoring.
    """
    def __init__(self, model_name="gpt2"):
        """
        Initialize the generative AI model.
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        print(f"Loading model: {model_name}")
        # Load the pre-trained tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load the pre-trained model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # Move model to the appropriate device
        self.model.to(device)
        # Set model to evaluation mode
        self.model.eval()
        print(f"Model loaded successfully on {device}")

    def generate_text(self, prompt, max_length=100, temperature=0.7, num_return_sequences=1):
        """
        Generate text based on a given prompt.
        Args:
            prompt (str): Input text to continue
            max_length (int): Maximum length of generated text (in tokens)
            temperature (float): Controls randomness; lower values are more deterministic)
            num_return_sequences (int): Number of different outputs to generate
        Returns:
            list: Generated text sequences
        """
        start_time = time.time()
        # Convert text prompt to tokens
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        # Generate text using the model
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,  # Enable sampling for more creative outputs
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2  # Prevent repetitive phrases
            )
        # Convert generated tokens back to text
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        inference_time = time.time() - start_time
        # Print performance metrics
        print(f"Generation completed in {inference_time:.2f} seconds")
        print(f"Generated {len(generated_texts)} sequence(s)")
        return generated_texts

    def get_model_info(self):
        """
        Get information about the loaded model.
        Returns:
            dict: Model information including parameters and configuration
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info = {
            "model_name": self.model.config.name_or_path,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "vocab_size": self.tokenizer.vocab_size,
            "device": str(device)
        }
        return info

if __name__ == "__main__":
    # Create an instance of the generative AI model
    print("Initializing Simple Generative AI Model...")
    gen_ai = SimpleGenerativeAI("gpt2")

    # Display model information
    model_info = gen_ai.get_model_info()
    print("Model Information:")
    print("-" * 50)
    for key, value in model_info.items():
        if "parameters" in key:
            # Format large numbers for better readability
            print(f"{key.replace('_', ' ').title()}: {value:,}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    print("-" * 50)

    # Test Case 1: Simple story beginning
    print("Test Case 1: Story Generation")
    print("=" * 60)
    prompt1 = "Once upon a time in a distant galaxy"
    generated_text1 = gen_ai.generate_text(
        prompt=prompt1,
        max_length=80,
        temperature=0.5,
        num_return_sequences=1
    )
    print(f"Prompt: {prompt1}")
    print(f"Generated: {generated_text1[0]}")
    print("\n" + "=" * 60 + "\n")

    # Test Case 2: Technical explanation
    print("Test Case 2: Technical Content")
    print("=" * 60)
    prompt2 = "Artificial intelligence is"
    generated_text2 = gen_ai.generate_text(
        prompt=prompt2,
        max_length=70,
        temperature=0.1,  # Lower temperature for more focused output
        num_return_sequences=1
    )
    print(f"Prompt: {prompt2}")
    print(f"Generated: {generated_text2[0]}")
    print("\n" + "=" * 60 + "\n")

    # Interactive testing - modify this prompt to test your own inputs
    custom_prompt = "Machine learning is an important skill in this job market"
    print(f"Testing custom prompt: '{custom_prompt}'")
    print("=" * 60)
    custom_result = gen_ai.generate_text(
        prompt=custom_prompt,
        max_length=100,          # Adjust length as needed
        temperature=0.5,         # Adjust creativity (0.1 = conservative, 1.0 = creative)
        num_return_sequences=1   # Number of different outputs
    )
    print(f"Input: {custom_prompt}")
    print(f"Output: {custom_result[0]}")
    print("\n" + "=" * 60)
    print("Model testing completed successfully!")
    print("The generative AI model is working and ready for deployment.")
    print("=" * 60)