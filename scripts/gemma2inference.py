from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")
model = AutoModelForCausalLM.from_pretrained("/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")

# 准备输入文本
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 生成输出
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# 解码输出文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)