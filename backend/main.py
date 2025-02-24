from fastapi import FastAPI, HTTPException, Depends
import asyncio
import aiohttp
import json
import numpy as np
from rank_bm25 import BM25Plus
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Text
import os
from pydantic_settings import BaseSettings
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройки базы данных
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

with open(os.path.join(settings.DATA_PATH, "processed_keywords.json"), "r", encoding="utf-8") as f:
    processed_keywords = json.load(f)

questions = [q["question"] for q in processed_keywords.values()]
keywords = [q["keywords"] for q in processed_keywords.values()]

with open(os.path.join(settings.DATA_PATH, "qna_dataset_smile_330_with_context.json"), "r", encoding="utf-8") as f:
    qna_data = json.load(f)

tokenized_questions = [q.lower().split() for q in questions]
bm25 = BM25Plus(tokenized_questions)

async def refine_query_with_llm(user_query: str):
    """LLM уточняет запрос пользователя."""
    data = {
        "model": settings.LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Ты помощник по документации. Преврати пользовательский вопрос в точный поисковый запрос. "
                           "Добавь ключевые слова, если необходимо, но не меняй смысл. "
                           "Выводи результат в кратком, удобном для поиска формате. "
                           "Пример:" 
                           "1. Вопрос: 'Как авторизоваться в системе?' → Поисковый запрос: 'Методы авторизации в системе SMILE'"
            },
            {
                "role": "user",
                "content": f"Преврати этот вопрос в точный поисковый запрос: '{user_query}'"
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.LLM_API_URL, json=data, headers=headers, timeout=30) as response:
                response_data = await response.json()
                refined_query = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return refined_query if refined_query and len(refined_query) >= 5 else user_query
    except aiohttp.ClientError as e:
        print(f"Ошибка при запросе к LLM: {e}")
        return user_query

async def rerank_candidates_with_llm(user_query: str, candidates: list):
    """LLM ранжирует найденные вопросы и выбирает лучший."""
    prompt = f"Какой из следующих вопросов наиболее похож на запрос: '{user_query}'?\n\n"
    prompt += "\n".join([f"{idx+1}. {q}" for idx, q in enumerate(candidates)])
    data = {
        "model": settings.LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Ты помогаешь находить наиболее релевантные вопросы. "
                           "Выбери наиболее подходящий вопрос из списка и выведи его номер."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.LLM_API_URL, json=data, headers=headers, timeout=30) as response:
                response_data = await response.json()
                best_match_idx = int(response_data.get("choices", [{}])[0].get("message", {}).get("content", "1").split(".")[0]) - 1
                return candidates[best_match_idx] if 0 <= best_match_idx < len(candidates) else candidates[0]
    except aiohttp.ClientError as e:
        print(f"Ошибка при запросе к LLM: {e}")
        return candidates[0]

def find_answer(matched_question):
    """ Находит ответ на вопрос в `qna_dataset_smile_330_with_context.json`."""
    for qna_id, qna_entry in qna_data.items():
        if matched_question in qna_entry["questions_list"]:
            index = qna_entry["questions_list"].index(matched_question)
            return qna_entry["answers_list"][index]
    return "Ответ не найден в базе."

@app.post("/ask")
async def ask_question(data: dict, db: AsyncSession = Depends(get_db)):
    user_query = data.get("question", "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Вопрос не должен быть пустым")
    
    refined_query = await refine_query_with_llm(user_query)
    tokenized_query = refined_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = np.argsort(bm25_scores)[-10:]
    best_candidates = [questions[i] for i in reversed(top_bm25_idx)]
    
    best_match = await rerank_candidates_with_llm(user_query, best_candidates)
    answer = find_answer(best_match)

    new_entry = QuestionAnswer(question=user_query, answer=answer)
    db.add(new_entry)
    await db.commit()
    
    return {"search_query": user_query, "matched_question": best_match, "answer":answer}

@app.on_event("startup")
async def startup():
    print("Инициализация БД...")
    await init_db()