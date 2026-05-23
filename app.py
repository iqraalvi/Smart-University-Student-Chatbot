from flask import Flask, render_template, request, jsonify
import pickle
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

index = faiss.read_index("chatbot_index.faiss")

with open("chatbot_dataset.pkl", "rb") as f:
    df = pickle.load(f)

model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")


def chatbot_response(question):
    question = question.lower().strip()

    if question == "":
        return "Please type a question."

    user_embedding = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(user_embedding)

    scores, indexes = index.search(user_embedding, k=1)

    best_score = float(scores[0][0])
    best_index = int(indexes[0][0])

    if best_score < 0.45:
        return "Sorry, I could not understand properly."

    return str(df.iloc[best_index]["answer"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    answer = chatbot_response(question)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    
    app.run(debug=False, use_reloader=False)