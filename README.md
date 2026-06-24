# AutoStatAI - Full Soccer Match Computer Vision Pipeline - Detection, Tracking, Events, Event To Player, Backend, Frontend
<img width="500" height="350" alt="FinalProject2_" src="https://github.com/user-attachments/assets/38505ed6-348f-444f-98f6-b472906b7bb8" />

## HighLevel Architecture
We developed our pipeline to be modular, 
<img width="1143" height="452" alt="preview" src="https://github.com/user-attachments/assets/9844ec68-592e-44b2-9f7d-9a996349f287" />

## [IMPORTANT] Limitations
Based on our test set gathered from real games, we reached to the limitations of our pipeline to maximize results quality:
- Requires a high-angle camera view similar to professional match broadcasts
- Distinct 2 jersey colors between teams
- Grass/synthetic field environments
- Remove noise from the frames: balls outside of field, people standing outside
- Some players that are out of the frame, and then theyre getting back to it, can get different ID than they had before.

## Requirements
- Nvidia RTX2080 GPU 24GB RAM
 
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

## Future Challenges
In order to make this pipeline more reliable, make sure to follow these steps:
- Improve ReID - players that are getting out of frame, and then getting back to it, should have the same ID as they had before
- Add calibration module: IT will improve tracking, can be used to improve events detection)
- Gather data, and then finetune model on low angle shoots
Note: We suggest to follow SoccerNet challenges that occure every year, and add/update new/existing modules.

