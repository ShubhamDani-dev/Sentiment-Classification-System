# Sentiment Classification System

A simple web app that tells you if text is positive or negative. Built this to learn more about ML models and how to integrate them into web applications.

## What it does

- Analyzes text and determines if it's positive or negative sentiment
- Has a web interface where you can type text and get results
- Also provides an API if you want to integrate it into other projects
- Uses a pre-trained RoBERTa model from Hugging Face (fancy transformer stuff)
- Can fall back to simple rule-based classification if needed

The whole thing runs in Docker containers so it's easy to set up and run anywhere.

## How it works

Frontend (React) talks to backend (FastAPI) which loads a machine learning model to analyze text.

```
React App (port 3000) → FastAPI (port 8000) → ML Model
```

Pretty straightforward setup.

## Getting started

You need Docker installed. That's it.

```bash
# Clone this repo
git clone https://github.com/ShubhamDani-dev/Sentiment-Classification-System.git
cd "Sentiment Classification System"

# Start everything
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

Then open http://localhost:3000 in your browser.

The API docs are at http://localhost:8000/docs if you want to see what endpoints are available.

## Using it

### Web interface
Just type some text and hit the button. You'll get back whether it's positive or negative with a confidence score.

### API
If you want to use it programmatically:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is awesome!"}'
```

You'll get back something like:
```json
{
  "label": "positive", 
  "score": 0.89
}
```

There are a few other endpoints:
- `/health` - check if everything is working
- `/model-status` - see which model is currently active
- `/enable-ml-model` - switch to the ML model (it starts with a simple rule-based one)
- `/debug-predict` - get more detailed info about the prediction

## Development

If you want to work on this locally without Docker:

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend  
```bash
cd frontend
npm install
npm start
```

The React dev server will proxy API calls to the backend automatically.

## Training your own model

There's a `finetune.py` script if you want to train on your own data. You'll need a JSONL file with text and labels:

```
{"text": "This is great!", "label": "positive"}
{"text": "This sucks", "label": "negative"}
```

Then:
```bash
python finetune.py --data your_data.jsonl --epochs 3
```

After training, restart the backend and it should pick up your new model.

## Docker stuff

```bash
# Start
docker-compose up -d

# Stop  
docker-compose down

# See logs
docker-compose logs

# Rebuild after changes
docker-compose build && docker-compose up -d
```

## Testing the API

FastAPI auto-generates docs at http://localhost:8000/docs which is pretty handy.

Some quick tests:

```bash
# Make sure it's alive
curl http://localhost:8000/health

# Try some text
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is pretty good"}'

# Try negative text  
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Ugh this sucks"}'
```

For the frontend, there's the usual:
```bash
cd frontend
npm test
```

Though honestly I haven't written comprehensive tests yet. PRs welcome.

## Configuration

You can tweak some settings by editing `backend/app/config.py`:

```python
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
USE_ML_MODEL = True
CONFIDENCE_THRESHOLD = 0.7
```

Or set environment variables if you want:
```env
BACKEND_PORT=8000
REACT_APP_API_URL=http://localhost:8000
```

## Deployment

For production you'd probably want to:
- Use a proper WSGI server instead of uvicorn
- Set up HTTPS 
- Use environment variables for secrets
- Maybe add some monitoring

I haven't included production configs yet but docker-compose should get you most of the way there.

## Project Structure

```
Sentiment Classification System/
├── README.md
├── docker-compose.yml
├── finetune.py                  # Training script
├── sample_data.jsonl            # Example training data
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py              # FastAPI app
│       ├── models.py            # Request/response models
│       └── sentiment.py         # The actual ML stuff
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js
│       ├── App.css
│       └── index.js
├── model/                       # Where trained models go
└── data/                        # For datasets
```

