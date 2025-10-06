from fastapi import FastAPI, Request
from agents import Agent, Runner, OpenAIChatCompletionsModel
from client import external_client
from dotenv import load_dotenv
import os
import requests
import uvicorn
import json

load_dotenv()

app = FastAPI(title="🌾 AgriGenius MCP Server (Gemini-Powered)")

FIREBASE_URL = os.getenv("FIREBASE_DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

# General agent
chatbot_agent = Agent(
    name="AgriGenius Chatbot Agent",
    instructions="You answer general agricultural questions about crops, fertilizers, and weather in a friendly way.",
    model="gemini-1.5-flash"
)

# Technical agent
agriculture_agent = Agent(
    name="Agriculture Expert Agent",
    instructions=(
        "You are an advanced agricultural expert. "
        "When a user asks about crop diseases, irrigation, soil health, or technical details, "
        "you respond ONLY in pure JSON format like:\n"
        "{'topic': 'Crop Disease Report', 'disease': 'Blight', 'solution': 'Use Mancozeb fungicide', 'irrigation_advice': 'Avoid overwatering'}"
    ),
    model="gemini-1.5-flash"
)

# Router agent
main_agent = Agent(
    name="Main Agent",
    instructions="Route queries: general → chatbot_agent, technical → agriculture_agent.",
    model="gemini-1.5-flash",
    handoffs=[chatbot_agent, agriculture_agent],
)

runner = Runner()


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

    try:
        result = await runner.run(main_agent, input=user_query)
        response_text = result.output_text.strip()
    except Exception as e:
        return {"error": f"Agent error: {str(e)}"}

    # Try to parse JSON from Gemini's response
    parsed_data = None
    try:
        parsed_data = json.loads(response_text.replace("'", "\""))
    except json.JSONDecodeError:
        # Fallback if model didn't follow format strictly
        parsed_data = {
            "topic": "General Response",
            "analysis": response_text
        }

    # Save in Firebase (optional)
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

    return parsed_data  # ✅ frontend expects direct JSON object


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
