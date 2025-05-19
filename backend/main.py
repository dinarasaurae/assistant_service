from fastapi import FastAPI, HTTPException, Depends
import asyncio
import aiohttp
import json
import numpy as np
import re
from typing import Any, List, Mapping
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Text
import os
from pydantic_settings import BaseSettings
from fastapi.middleware.cors import CORSMiddleware
from fuzzywuzzy import fuzz
import difflib
from nltk.stem.snowball import RussianStemmer
import faiss
import rank_bm25

class Settings(BaseSettings):
    APP_NAME: str = "Chat Assistant"
    APP_VERSION: str = "1.0.0"
    DB_USER: str = os.getenv("DB_USER", "user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "password")
    DB_NAME: str = os.getenv("DB_NAME", "chat_db")
    DB_HOST: str = os.getenv("DB_HOST", "db")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DATABASE_URL: str = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    LLM_API_URL: str = os.getenv("LLM_API_URL", "http://10.32.15.90:8007/v1/chat/completions")
    LLM_MODEL: str = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    DATA_PATH: str = "data/"

settings = Settings()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

engine = create_async_engine(settings.DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class QuestionAnswer(Base):
    __tablename__ = "questions_answers"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with SessionLocal() as session:
        yield session

stemmer = RussianStemmer()
vectorizer = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def normalize(text):
    return text.strip().lower().replace("ё", "е")

def clean_text(text):
    text = normalize(text)
    noise_phrases = [
        "на платформе smile", "в системе smile", "в сервисе smile", "на сервисе smile", "в smile", "платформе smile",
        "в приложении smile", "в среде smile", "сервисе smile", "на smile", "платформы smile", "smile", "SMILE"
    ]
    for phrase in noise_phrases:
        text = text.replace(phrase, "")
    text = re.sub(r"\\s+([?.!,])", r"\\1", text)
    text = re.sub(r"\\s{2,}", " ", text)
    return text.strip()

def jaccard_similarity(a, b):
    tokens1 = set(stemmer.stem(w) for w in a.split())
    tokens2 = set(stemmer.stem(w) for w in b.split())
    return len(tokens1 & tokens2) / len(tokens1 | tokens2) if tokens1 | tokens2 else 0

def soft_match_score(q1: str, q2: str) -> float:
    jaccard = jaccard_similarity(q1, q2)
    sequence = difflib.SequenceMatcher(None, q1, q2).ratio()
    fuzzy = fuzz.ratio(q1, q2) / 100.0
    return (0.3 * jaccard) + (0.3 * sequence) + (0.4 * fuzzy)

with open(os.path.join(settings.DATA_PATH, "processed_keywords1.json"), "r", encoding="utf-8") as f:
    processed_keywords_raw = json.load(f)
with open(os.path.join(settings.DATA_PATH, "qna_dataset_smile_330_with_context.json"), "r", encoding="utf-8") as f:
    qna_dataset = json.load(f)
with open(os.path.join(settings.DATA_PATH, "synonymous_questions.json"), "r", encoding="utf-8") as f:
    synonyms_data = json.load(f)

question_keywords = {clean_text(v["question"]): v["keywords"] for v in processed_keywords_raw.values()}
question_answer_pairs = [(clean_text(q), a) for v in qna_dataset.values() for q, a in zip(v["questions_list"], v["answers_list"])]
synonym_to_canonical = {}
for v in synonyms_data.values():
    canon = clean_text(v["question"])
    for syn in v.get("synonymous_questions", []):
        synonym_to_canonical[clean_text(syn.strip('" '))] = canon

question_vectors = np.array([vectorizer.encode(q) for q, _ in question_answer_pairs])
vector_dim = question_vectors.shape[1]
faiss_index = faiss.IndexFlatIP(vector_dim)
faiss_index.add(question_vectors)
bm25 = rank_bm25.BM25Plus([q.split() for q, _ in question_answer_pairs])

async def llm_complete(prompt: str) -> str:
    data = {"model": settings.LLM_MODEL, "messages": [{"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.LLM_API_URL, json=data, headers=headers, timeout=30) as response:
                response_data = await response.json()
                return response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Ошибка запроса LLM: {e}")
        return prompt

async def refine_query(query: str) -> str:
    query_clean = clean_text(query)
    if query_clean in synonym_to_canonical:
        base = synonym_to_canonical[query_clean]
    else:
        base = max(question_keywords, key=lambda q: cosine_similarity(
            vectorizer.encode(query_clean).reshape(1, -1), vectorizer.encode(q).reshape(1, -1))[0][0])
    keywords = question_keywords.get(base, "")
    prompt = f'Переформулируй вопрос, добавив ключевые слова: {keywords}. Вопрос: "{query}"'
    return clean_text(await llm_complete(prompt))

def find_best_match(refined_query: str):
    query_clean = clean_text(refined_query)
    query_vector = vectorizer.encode(query_clean).reshape(1, -1)
    bm25_scores = bm25.get_scores(query_clean.split())
    hybrid_scores = {
        i: 0.2 * bm25_scores[i] + 0.8 * cosine_similarity(query_vector, question_vectors[i].reshape(1, -1))[0][0]
        for i in range(len(question_vectors))
    }
    best = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    for i, _ in best:
        candidate = question_answer_pairs[i][0]
        sim = float(cosine_similarity(query_vector, question_vectors[i].reshape(1, -1))[0][0])
        soft_score = soft_match_score(query_clean, candidate)
        if sim > 0.65 or soft_score > 0.8:
            return candidate, question_answer_pairs[i][1]
    return None, None

def find_answer(matched_question):
    for qna_entry in qna_dataset.values():
        if matched_question in qna_entry["questions_list"]:
            index = qna_entry["questions_list"].index(matched_question)
            return qna_entry["answers_list"][index]
    return "Ответ не найден."

@app.post("/ask")
async def ask_question(data: dict, db: AsyncSession = Depends(get_db)):
    user_query = data.get("question", "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Вопрос не должен быть пустым")
    refined = await refine_query(user_query)
    matched_q, answer = find_best_match(refined)
    if not answer:
        answer = await llm_complete(user_query)
        matched_q = None
    db.add(QuestionAnswer(question=user_query, answer=answer))
    await db.commit()
    return {"search_query": user_query, "matched_question": matched_q, "answer": answer}

@app.on_event("startup")
async def startup_event():
    await init_db()