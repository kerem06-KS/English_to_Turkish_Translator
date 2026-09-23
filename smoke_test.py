from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Machine translation is difficult for ancient languages."

inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)