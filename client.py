# main.py
from fastapi import FastAPI, Request
from agents import Agent, Runner, OpenAIChatCompletionsModel
from client import external_client
from dotenv import load_dotenv
import os
import requests
import uvicorn
import json

# Load environment variables
load_dotenv()

# Ensure API keys are set
if not os.getenv("OPENAI_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("GEMINI_API_KEY")

FIREBASE_URL = os.getenv("FIREBASE_URL")  # Optional Firebase integration

app = FastAPI(title="🌾 AgriGenius MCP Server (Gemini-Powered)")

# --- Model setup ---
model = OpenAIChatCompletionsModel(
    model="google/gemini-1.5-flash",
    openai_client=external_client
)

# --- Agents ---
chatbot_agent = Agent(
    name="AgriGenius Chatbot Agent",
    instructions="You answer general agricultural questions about crops, fertilizers, and weather in a friendly way.",
    model="google/gemini-1.5-flash"
)

agriculture_agent = Agent(
    name="Agriculture Expert Agent",
    instructions=(
        "You are an advanced agricultural expert. "
        "When a user asks about crop diseases, irrigation, soil health, or technical details, "
        "you respond ONLY in pure JSON format like:\n"
        "{'topic': 'Crop Disease Report', 'disease': 'Blight', 'solution': 'Use Mancozeb fungicide', 'irrigation_advice': 'Avoid overwatering'}"
    ),
    model="google/gemini-1.5-flash"
)

main_agent = Agent(
    name="Main Agent",
    instructions="Route queries: general → chatbot_agent, technical → agriculture_agent.",
    model="google/gemini-1.5-flash",
    handoffs=[chatbot_agent, agriculture_agent],
)

runner = Runner()

# --- Routes ---
@app.get("/")
def home():
    return {"message": "🌾 AgriGenius MCP Server is running with Gemini!"}

@app.post("/query")
async def process_query(request: Request):
    """
    Process user query and route to appropriate agent.
    """
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON body."}

    user_query = data.get("query")
    user_id = data.get("user_id", "anonymous")

    if not user_query:
        return {"error": "Missing 'query' in request body."}

    try:
        # Run the agent synchronously
        result = runner.run(main_agent, input=user_query)
        response_text = result.output_text.strip()
    except Exception as e:
        return {"error": f"Agent error: {str(e)}"}

    # Try parsing the agent response as JSON
    try:
        parsed_data = json.loads(response_text.replace("'", "\""))
    except json.JSONDecodeError:
        parsed_data = {"topic": "General Response", "analysis": response_text}

    # Optional: save to Firebase
    if FIREBASE_URL:
        try:
            payload = {"user_id": user_id, "query": user_query, "response": parsed_data}
            requests.post(f"{FIREBASE_URL}/agriReports.json", json=payload)
        except Exception as e:
            print("⚠️ Firebase Save Error:", e)

    return parsed_data

# --- Run server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
