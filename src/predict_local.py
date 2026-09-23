from transformers import pipeline

clf = pipeline("text-classification", model="models/final", device=0)
for text in ["I was charged twice for my card payment",
             "How do I reset my PIN?",
             "My card still hasn't arrived after two weeks"]:
    print(text, "->", clf(text))
