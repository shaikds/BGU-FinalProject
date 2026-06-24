# AutoStatAI - A Full Soccer Match Computer Vision Pipeline - Detection, Tracking, Events

## Preview
https://github.com/user-attachments/assets/540edbd1-3164-40c9-99d7-1bac47bb9768


## HighLevel Architecture
We developed our 
<img width="1142" height="482" alt="image" src="https://github.com/user-attachments/assets/f02c25cd-4dbe-476f-84ba-84c2b2d8e04a" />





## [IMPORTANT] Limitations

Based on our test set gathered from real games, we reached to the limitations of our pipeline to maximize results quality:
- Requires a high-angle camera view similar to professional match broadcasts
- Distinct 2 jersey colors between teams
- Grass/synthetic field environments
- Remove noise from the frames: balls outside of field, people standing outside
- Nvidia 
 
## Quickstart
1. Clone this repo
2. Checkout detection, events modules, get in to their source repos to download each module weights
3. Setup the frontend and backend
4. Input your video through run_both.py or orchestrator.py
5. [Optional] To visualize your results as in our preview here, use unified_visualizer.py

## FrontEnd/BackEnd Quick Start

### 1. MongoDB
docker run -d -p 27017:27017 --name mongo mongo
### If already created the mongoDB container before, run this instead:
docker start mongo

### 2. Backend
cd backend
conda activate soccer_backend   # חשוב!
uvicorn main:app --reload

### 3. Frontend
cd frontend
pnpm dev


