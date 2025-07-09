# LiveAI Backend

## Setup

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```sh
   uvicorn backend.main:app --reload
   ```

The server will be available at http://localhost:8000

## API

- `POST /predict` — Upload an image for detection and activity recognition.

---

Once the model is integrated, the backend will return detected people, their activities, and confidence scores. # live-ai-activity
