import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/share_docker/DeepSeek-R1-0528-Qwen3-8B"
device = "cuda"

try:
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16)
    print("Model loaded.")

    prompt = "上下文：这是一个测试上下文。\n问题：这是一个测试问题。\n答案："
    print(f"Prompt: {prompt}")
    
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Input IDs length: {enc.input_ids.shape[1]}")
    
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=100, do_sample=False)
    
    print(f"Output IDs length: {out.shape[1]}")
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Generated text: {text}")
    
    if "答案：" in text:
        ans = text.split("答案：", 1)[-1].strip()
        print(f"Extracted answer: '{ans}'")
    else:
        print("Could not find '答案：' in output.")

except Exception as e:
    print(f"Error: {e}")
