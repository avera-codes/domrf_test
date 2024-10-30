import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    AutoModelForQuestionAnswering,
)
import pandas as pd
import faiss
import nltk
import re
import pickle
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab", quiet=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модели и токенайзеры
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)

instruction_model_name = "google/flan-t5-base"
instruction_tokenizer = AutoTokenizer.from_pretrained(instruction_model_name)
instruction_model = T5ForConditionalGeneration.from_pretrained(
    instruction_model_name
).to(device)

qa_model_name = "deepset/roberta-base-squad2"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(device)

# Загрузка и подготовка данных
df = pd.read_csv("papers.csv").dropna(subset=["Title", "Text"])


# Кэширование эмбеддингов
def compute_embeddings(texts):
    """Создает и кэширует эмбеддинги, если их нет."""
    cached_embeddings_path = "embeddings_cache.pkl"
    try:
        with open(cached_embeddings_path, "rb") as f:
            sentence_embeddings = pickle.load(f)
    except FileNotFoundError:
        sentence_embeddings = get_embeddings(texts)
        with open(cached_embeddings_path, "wb") as f:
            pickle.dump(sentence_embeddings, f)
    return sentence_embeddings


def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = embedding_tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings)


# Функция токенизации
def split_and_tokenize(text):
    blocks = text.split("\n\n")
    sentences = []
    for block in blocks:
        tokenized = nltk.sent_tokenize(block)
        combined = []
        i = 0
        while i < len(tokenized):
            if re.match(r"^\d+\.?$", tokenized[i]):
                if i + 1 < len(tokenized):
                    combined.append(f"{tokenized[i]} {tokenized[i + 1]}")
                    i += 2
                else:
                    combined.append(tokenized[i])
                    i += 1
            else:
                combined.append(tokenized[i])
                i += 1
        sentences.extend(combined)
    return sentences


df["sentences"] = df["Text"].apply(split_and_tokenize)
sentences = df.explode("sentences").reset_index(drop=True)
sentence_embeddings = compute_embeddings(sentences["sentences"].tolist()).numpy()

# Индексирование FAISS
index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
index.add(sentence_embeddings)


# Функции для поиска и генерации ответа
def search_relevant_sentences(question, k=5):
    question_embedding = get_embeddings([question]).detach().numpy()
    distances, indices = index.search(question_embedding, k)
    return sentences.iloc[indices[0]]["sentences"].tolist()


def extract_answer(question, context):
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True).to(
        device
    )
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
    )
    return answer.strip(), outputs.start_logits.max().item()


def format_answer(answer):
    return answer[0].upper() + answer[1:] if answer else answer


def answer_question(question):
    if any(
        word in question.lower()
        for word in ["how", "steps", "procedure", "process", "implement", "deploy"]
    ):
        relevant_sentences = search_relevant_sentences(question, k=3)
        combined_context = " ".join(relevant_sentences)
        input_text = f"Summarize the steps for {question}. {combined_context}"
        inputs = instruction_tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = instruction_model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
            )
        return format_answer(
            instruction_tokenizer.decode(outputs[0], skip_special_tokens=True)
        )

    relevant_sentences = search_relevant_sentences(question)
    answers = [extract_answer(question, sentence) for sentence in relevant_sentences]
    filtered_answers = [ans for ans in answers if ans[0]]
    if not filtered_answers:
        return "Ответ не найден"
    best_answer = max(filtered_answers, key=lambda x: x[1])
    return format_answer(best_answer[0])


# Streamlit UI
st.title("Question-Answering System")
question = st.text_input("Введите ваш вопрос:")
if st.button("Получить ответ"):
    if question:
        answer = answer_question(question)
        st.write("Ответ:", answer)
    else:
        st.write("Пожалуйста, введите вопрос.")
