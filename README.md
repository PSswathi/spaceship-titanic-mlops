# 1. Build the image
docker build -t spaceship-titanic-app:latest .

# 2. Run the container
docker run -d -p 8000:8000 --name spaceship-titanic-app spaceship-titanic-app:latest

# 3. Test it
curl http://localhost:8000/health

# 4. Check logs
docker logs spaceship-titanic-app


Step 1 — Create & activate venv
bashcd /Users/swathi/Downloads/spaceship-titanic/spaceship-titanic-mlops
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
Step 2 — Run data loader
bashpython src/data_loader.py
Step 3 — Run feature engineering
bashpython src/feature_engg.py
Step 4 — Start MLflow UI (keep this terminal open)
bashmlflow ui --backend-store-uri mlruns --port 5001

http://127.0.0.1:5001


