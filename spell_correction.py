from transformers import pipeline

corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2", device=0)


# Example
MAX_LENGTH = 512

# Define the text samples
texts = [
    "Coong hòa xax hội chủ nghĩa Việt Nam, độc ập tự do hạnh phúc"
]

# Batch prediction
predictions = corrector(texts, max_length=MAX_LENGTH)

# Print predictions
for text, pred in zip(texts, predictions):
    print("- " + pred['generated_text'])


from underthesea import text_normalize
print(text_normalize("Coong hòa xax hội chủ nghĩa Việt Nam, độc ập tự do hạnh phúc"))
