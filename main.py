# main.py
from fastapi import FastAPI, Request
from agents import Agent, Runner, OpenAIChatCompletionsModel
from client import external_client  # your shared Gemini client
from dotenv import load_dotenv
import os
import requests
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="🌾 AgriGenius MCP Server (Gemini-Powered)")

# -------------------------
#  ENV VARIABLES
# -------------------------
FIREBASE_URL = os.getenv("FIREBASE_DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------
#  MODEL & AGENTS
# -------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

# Chatbot Agent: general help
chatbot_agent = Agent(
    name="AgriGenius Chatbot Agent",
    instructions="You answer general agricultural questions about crops, fertilizers, and weather in a friendly way.",
    model="gemini-1.5-flash"
)

# Agriculture Agent: expert-level
agriculture_agent = Agent(
    name="Agriculture Expert Agent",
    instructions=(
        "You are an advanced agricultural expert. "
        "When a user asks about crop diseases, irrigation, soil health, or technical details, "
        "you give structured responses in JSON format like:\n"
        "{'topic': 'Crop Disease Report', 'disease': 'Blight', 'solution': 'Use Mancozeb fungicide', 'irrigation_advice': 'Avoid overwatering'}"
    ),
    model="gemini-1.5-flash"
)

# Main agent: router
main_agent = Agent(
    name="Main Agent",
    instructions="Route user queries. If general, use chatbot_agent; if technical or about crops, use agriculture_agent.",
    model="gemini-1.5-flash",
    handoffs=[chatbot_agent, agriculture_agent],
)

runner = Runner()

# -------------------------
#  ENDPOINTS
# -------------------------
@app.get("/")
def home():
    return {"message": "🌾 AgriGenius MCP Server is running with Gemini!"}


@app.post("/query")
async def process_query(request: Request):
    """
    Handles incoming crop or chat queries from Firebase frontend.
    Runs the main agent, and stores results in Firebase DB.
    """
    data = await request.json()
    user_query = data.get("query")
    user_id = data.get("user_id", "anonymous")

    if not user_query:
        return {
            "data": None,
            "error": "Missing 'query' in request body."
        }

    try:
        # Run multi-agent flow
        result = await runner.run(main_agent, input=user_query)
        response_text = result.output_text
    except Exception as e:
        return {
            "data": None,
            "error": f"Agent error: {str(e)}"
        }

    # Optional: store in Firebase Realtime DB
    if FIREBASE_URL:
        try:
            payload = {
                "user_id": user_id,
                "query": user_query,
                "response": response_text,
            }
            requests.post(f"{FIREBASE_URL}/agriReports.json", json=payload)
        except Exception as e:
            print("⚠️ Firebase Save Error:", e)

    # ✅ Structured JSON your frontend expects
    return {
        "data": {
            "analysis": response_text
        },
        "error": None
    }


# -------------------------
#  SERVER START (for Railway)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
