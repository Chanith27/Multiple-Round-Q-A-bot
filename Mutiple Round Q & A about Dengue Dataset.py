import pandas as pd
from transformers import pipeline

# Load your dengue dataset
df = pd.read_csv(r'D:\Apex Protocol\Data Science & Mathermatics\AI Model Project\Dengue_Data.csv')

# Show column names to help you know what to use
print("📌 Dataset Columns:", df.columns.tolist())

# 🔁 Convert dataset into a paragraph-like context
context = "\n".join([
    f"In {row['Date']}, there were {row['City']} city and {row['Value']} value."
    for index, row in df.iterrows()
])

# Load the Hugging Face Question Answering pipeline
qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

print("\n💬 Type your question about the dataset (type 'exit' to quit):\n")

# 🧠 Chat loop
while True:
    question = input("You: ")
    if question.lower() in ['exit', 'quit']:
        print("👋 Exiting... Bye!")
        break
    
    result = qa(question=question, context=context)
    print("Bot:", result['answer'])
