import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load FAQ data
with open("faq_data.json", "r") as file:
    faq_data = json.load(file)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]
# Convert questions into TF-IDF vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

print("===== FAQ Chatbot =====")
print("Type 'exit' to quit.\n")

while True:
    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("Chatbot: Thank you! Have a nice day.")
        break

    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()

    if similarity[0][best_match] > 0.2:
        print("Chatbot:", answers[best_match])
    else:
        print("Chatbot: Sorry, I don't know the answer to that question.")