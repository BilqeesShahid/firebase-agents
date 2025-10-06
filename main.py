# main.py
from fastapi import FastAPI, Request
from agents import Agent, Runner, OpenAIChatCompletionsModel
from client import external_client
from dotenv import load_dotenv
import os
import requests
import uvicorn
import json

# Gemini service account imports
from google.oauth2 import service_account
import google.auth.transport.requests

load_dotenv()

# -------------------
# Environment Variables
# -------------------
FIREBASE_URL = os.getenv("FIREBASE_DATABASE_URL")
SERVICE_ACCOUNT_JSON = os.getenv("SERVICE_ACCOUNT_JSON")
GEMINI_PROJECT_ID = os.getenv("GEMINI_PROJECT_ID")
PORT = int(os.getenv("PORT", 8080))

# -------------------
# Setup Gemini Credentials
# -------------------
# Write SERVICE_ACCOUNT_JSON to a temp file
with open("service-account.json", "w") as f:
    f.write(SERVICE_ACCOUNT_JSON)

SERVICE_ACCOUNT_FILE = "service-account.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Helper function to call Gemini
def gemini_generate(prompt: str) -> str:
    credentials.refresh(google.auth.transport.requests.Request())
    token = credentials.token
    endpoint = f"https://generativelanguage.googleapis.com/v1/projects/{GEMINI_PROJECT_ID}/locations/global/models/gemini-2.5-flash-lite:generateText"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    body = {
        "prompt": {"text": prompt},
        "temperature": 0.7,
        "candidate_count": 1
    }
    resp = requests.post(endpoint, headers=headers, json=body)
    resp_json = resp.json()
    try:
        return resp_json["candidates"][0]["output"]
    except Exception:
        return f"Error from Gemini: {resp_json}"

# -------------------
# FastAPI setup
# -------------------
app = FastAPI(title="🌾 AgriGenius MCP Server (Gemini-Powered)")

# OpenAI model (if you want OpenAI fallback)
model = OpenAIChatCompletionsModel(
    model="models/text-bison-001",
    openai_client=external_client
)

# -------------------
# Agents
# -------------------
chatbot_agent = Agent(
    name="AgriGenius Chatbot Agent",
    instructions="You answer general agricultural questions about crops, fertilizers, and weather in a friendly way.",
    model=model
)

agriculture_agent = Agent(
    name="Agriculture Expert Agent",
    instructions=(
        "You are an advanced agricultural expert. "
        "When a user asks about crop diseases, irrigation, soil health, or technical details, "
        "you respond ONLY in pure JSON format like:\n"
        "{'topic': 'Crop Disease Report', 'disease': 'Blight', 'solution': 'Use Mancozeb fungicide', 'irrigation_advice': 'Avoid overwatering'}"
    ),
    model=model
)

main_agent = Agent(
    name="Main Agent",
    instructions="Route queries: general → chatbot_agent, technical → agriculture_agent.",
    model=model,
    handoffs=[chatbot_agent, agriculture_agent],
)

runner = Runner()

# -------------------
# Routes
# -------------------
@app.get("/")
def home():
    return {"message": "🌾 AgriGenius MCP Server is running with Gemini!"}

@app.post("/query")
async def process_query(request: Request):
    data = await request.json()
    user_query = data.get("query")
    user_id = data.get("user_id", "anonymous")

    if not user_query:
        return {"error": "Missing 'query' in request body."}

    # -------------------
    # Run agent
    # -------------------
    try:
        # Call Gemini directly instead of OpenAI SDK if you prefer
        response_text = gemini_generate(user_query).strip()
    except Exception as e:
        return {"error": f"Agent error: {str(e)}"}

    # -------------------
    # Parse JSON from Gemini response
    # -------------------
    try:
        parsed_data = json.loads(response_text.replace("'", "\""))
    except json.JSONDecodeError:
        parsed_data = {
            "topic": "General Response",
            "analysis": response_text
        }

    # -------------------
    # Optional Firebase saving
    # -------------------
    if FIREBASE_URL:
        try:
            payload = {
                "user_id": user_id,
                "query": user_query,
                "response": parsed_data
            }
            requests.post(f"{FIREBASE_URL}/agriReports.json", json=payload)
        except Exception as e:
            print("⚠️ Firebase Save Error:", e)

    return parsed_data

# -------------------
# Run locally
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


