import transformers
import torch
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token="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",
)

messages = [
    {"role": "system", "content": "You are a Legal Assistant.Focused on South African Law. And only South African Law.You are a helpful assistant.You answer questions about South African Law,specifically civil law,contract law, and family law."},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
