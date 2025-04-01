# LLM FastAPI Сервис
![image](https://github.com/user-attachments/assets/48cd2d22-28d2-4166-bbb0-456e819804d7)

## Описание проекта
Данный проект представляет собой **вопросно-ответный сервис**, построенный на базе **LLM**, FastAPI и Angular. Система позволяет пользователям получать ответы на вопросы по эксплуатации сервиса, используя обработку естественного языка, поиск по базе знаний и проксирование запросов через Nginx.

### Основные компоненты:
1. **Бэкенд (FastAPI)** — отвечает за обработку запросов, взаимодействие с базой данных и использование LLM.
2. **Фронтенд (Angular 19 + Vite)** — веб-интерфейс, с которым взаимодействует пользователь.
3. **Nginx** — проксирование запросов и статика.
4. **База данных (PostgreSQL)** — хранение вопросов и ответов.

---
## Структура проекта
```
ai_chat
├──  backend              # Бэкенд (FastAPI)
│   ├── Dockerfile.backend  # Dockerfile для бэкенда
│   ├── main.py             # Основной код API
│   ├── requirements.txt    # Зависимости Python
│   ├── .env                # Конфигурация окружения (БД, API и т. д.)
├──  frontend             # Фронтенд (Angular 19)
│   ├── Dockerfile.frontend # Dockerfile для фронтенда
│   ├── src/                # Исходный код приложения
│   ├── package.json        # Зависимости Node.js
│   ├── vite.config.ts      # Конфигурация Vite
├──  nginx                # Конфигурация Nginx
│   ├── nginx.conf          # Настройки проксирования
├── docker-compose.yml      # Контейнеризация всех сервисов
├── docker-compose.override.yml # Оверрайд для dev-режима
└── README.md               # Данный файл
```
---
## Установка и запуск

### 1. **Dev-режим** (локальная разработка)
```sh
git clone https://github.com/dinarasaurae/assistant_service.git
cd assistant_service
cp backend/.env.example backend/.env  # Настроить переменные окружения
cp frontend/.env.example frontend/.env

# Запуск контейнеров (оставляя бекенд отдельно запущенным на сервере)
docker-compose -f docker-compose.override.yml up --build
```
Фронт будет доступен по `http://localhost:4200` (позже добавлю prod), бекенд уже работает на сервере.

### 2. **Prod-режим** (развертывание)
```sh
# Запуск всех контейнеров
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
docker-compose up --build -d
```
Nginx сервирует фронт на `http://94.126.205.209/`.

---
## API-документация (FastAPI)

### **Запрос ответа**
**POST /ask**
- **Тело запроса**:
  ```json
  {"question": "Как авторизоваться в системе через Яндекс?"}
  ```
- **Пример ответа**:
  ```json
  {
    "search_query": "Как авторизоваться в системе через Яндекс?",
    "matched_question": "Какие действия необходимо выполнить для авторизации через социальную сеть Яндекс?",
    "answer": "Для авторизации через Яндекс, нажмите на кнопку Яндекс..."
  }
  ```

Полная документация API доступна по адресу: `http://94.126.205.209:8001/docs`

---
## Переменные окружения

**backend/.env:**
```
DB_USER=user
DB_PASSWORD=password
DB_NAME=chat_db
DB_HOST=db
DB_PORT=5432
LLM_API_URL=http://XXXXXXXXXX
```

**frontend/.env:**
```
VITE_API_URL=http://94.126.205.209:8001
```

---
## Планы на будущее
- Улучшение поиска и векторизации запросов
- Оптимизация работы с LLM
- Интеграция с дополнительными базами знаний

**Разработка ведётся! Если есть предложения — пишите!**

