# Quick Start
## 1. MongoDB (חייב לרוץ)
docker run -d -p 27017:27017 --name mongo mongo
## If already created the mongoDB container before, run this instead:
docker start mongo

## 2. Backend
cd backend
conda activate soccer_backend   # חשוב!
uvicorn main:app --reload

## 3. Frontend
cd frontend
pnpm dev
