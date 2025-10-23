"""
Simple Donut Demo - No user input required
Just run and see the result!
"""

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
from pdf2image import convert_from_path


def simple_donut_demo():
    """Simple demo that processes documents."""
    
    print("Simple Donut Model Demo")
    print("=" * 40)
    
    # Load model
    print("Loading Donut model...")
    model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
    
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}")
    
    # Load document
    print("Loading document...")
    
    try:
        # Try to load an image directly
        image = Image.open("test_dominos.jpeg").convert('RGB')
        print("Document loaded successfully")
    except Exception as e:
        print(f"Error loading document: {e}")
        return
    
    # Process the document
    print("Processing document...")
    
    # Prepare inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Ask question
    question = "what is the right side sauce?"
    task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    
    decoder_input_ids = processor.tokenizer(
        task_prompt, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).input_ids.to(device)
    
    # Generate answer
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # Decode result
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    
    # Extract just the answer part
    if "<s_answer>" in sequence:
        result = sequence.split("<s_answer>")[1].strip()
        result = re.sub(r"</s_answer>.*", "", result).strip()
    else:
        result = re.sub(r"<.*?>", "", sequence, count=1).strip()
    
    # Show results
    print("\n" + "=" * 40)
    print("QUESTION:", question)
    print("ANSWER:", result)
    print("=" * 40)
    
    return result


if __name__ == "__main__":
    try:
        simple_donut_demo()
    except Exception as e:
        print(f"Error: {str(e)}")
