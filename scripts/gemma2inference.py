from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf",
    device_map="cuda:0",
    load_in_8bit=False,
    torch_dtype=torch.float32
)

input_text = ["Write me a poem about Machine Learning.", "I want to eat a", "I go to the", "I want", "I"]
input_ids = tokenizer.batch_encode_plus(input_text, return_tensors="pt", padding='longest')

for t in input_ids:
    if torch.is_tensor(input_ids[t]):
        input_ids[t] = input_ids[t].to("cuda:0")

outputs = model.generate(**input_ids)
print(outputs)

print(tokenizer.batch_decode(outputs))