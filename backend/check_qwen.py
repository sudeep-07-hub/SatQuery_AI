import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Qwen/Qwen3-4B-Instruct-2507"
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True)
    print("Model loaded successfully!")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("mps")
    print("Generating...")
    outputs = model.generate(**inputs, max_new_tokens=10)
    print("Response:", tokenizer.decode(outputs[0][len(inputs.input_ids[0]):]))
except Exception as e:
    print("Error:", e)
