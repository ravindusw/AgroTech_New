from typing import Optional, Dict
from app.crop_recommendation import CropRecommendationSystem
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Query
from app.train import initial_train_model, retrain_model
from app.predict import predict_future_prices
from langchain_core.messages import AIMessage, HumanMessage
from app.agent import chat_app
from langchain.schema import BaseMessage  # optional import for clarity
from pydantic import BaseModel
from typing import List
import uuid
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AgroTech AI Backend",
    description="AI-powered backend for AgroTech",
    version="1.0.0"
)

# Enhanced CORS configuration for production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
# Remove empty strings and strip whitespace
allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]

# If no origins specified, use a restrictive default
if not allowed_origins:
    allowed_origins = ["http://localhost:3000", "http://localhost:3001", "https://agrotechui.onrender.com", "https://agrotech-1-tbst.onrender.com"]
    logging.warning("No ALLOWED_ORIGINS specified. Using localhost defaults.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# CORS headers middleware
# @app.middleware("http")
# async def add_cors_headers(request: Request, call_next):
#     response = await call_next(request)
#     return response

# # Handle preflight OPTIONS requests for chat endpoint
# @app.options("/chat")
# async def chat_preflight():
#     return {
#         "message": "OK"
#     }

# @app.options("/chat/clear")
# async def chat_clear_preflight():
#     return {
#         "message": "OK"
#     }

# Initialize crop recommendation system
crop_system = CropRecommendationSystem()

# Pydantic models for request/response
class CropRecommendationRequest(BaseModel):
    location: str
    use_defaults: Optional[bool] = False

class ManualCropRequest(BaseModel):
    nitrogen: float
    temperature: float
    ph: float
    rainfall: float
    humidity: float

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.get("/")
def root():
    return {"message": "AgroTech API - Rice Forecast & Crop Recommendation 🚀"}

# Existing rice forecasting endpoints
@app.post("/train")
def train():
    """Initial training from historical data"""
    initial_train_model()
    return {"status": "Model trained and saved successfully."}

@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    """Retrain from new data file (CSV)"""
    retrain_model(file)
    return {"status": "Model retrained with new data and saved."}

@app.get("/predict")
def predict():
    forecast_df, history_df = predict_future_prices(n_days=60)
    return {
        "forecast": forecast_df.to_dict(orient="records"),
        "history": history_df.to_dict(orient="records")
    }


# New crop recommendation endpoints
@app.post("/crop-recommendation")
def get_crop_recommendation(request: CropRecommendationRequest):
    """
    Get crop recommendations based on location data
    
    Fetches soil, weather, and rainfall data automatically and provides 
    crop recommendations based on current environmental conditions.
    """
    try:
        result = crop_system.get_crop_recommendations(
            location=request.location,
            use_defaults=request.use_defaults
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting crop recommendations: {str(e)}")

@app.post("/crop-recommendation/manual")
def get_manual_crop_recommendation(request: ManualCropRequest):
    """
    Get crop recommendations based on manually provided environmental data
    
    Uses the ML model to provide crop recommendations based on the 5 input features:
    N, temperature, humidity, pH, and rainfall.
    """
    try:
        # Prepare model input
        model_input = {
            'N': request.nitrogen,
            'temperature': request.temperature,
            'ph': request.ph,
            'rainfall': request.rainfall,
            'humidity': request.humidity
        }
        
        # Get recommendations from ML model
        recommendations = crop_system._predict_crops(model_input)
        
        return {
            'success': True,
            'input_data': {
                'nitrogen': request.nitrogen,
                'temperature': request.temperature,
                'ph': request.ph,
                'rainfall': request.rainfall,
                'humidity': request.humidity
            },
            'recommendations': recommendations,
            'model_used': 'LightGBM Classifier',
            'total_crops': len(crop_system.crop_labels)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting manual crop recommendations: {str(e)}")

@app.get("/crop-recommendation/details/{crop_name}")
def get_crop_details(crop_name: str):
    """Get detailed information about a specific crop"""
    try:
        details = crop_system.get_crop_details(crop_name)
        return {
            'success': True,
            'crop': crop_name,
            'details': details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting crop details: {str(e)}")

@app.get("/crop-recommendation/test")
def test_crop_system():
    """Test the crop recommendation system with sample data"""
    try:
        # Test with Matale, Sri Lanka as default
        result = crop_system.get_crop_recommendations("Matale, Sri Lanka", use_defaults=True)
        return {
            'success': True,
            'message': 'Crop recommendation system test',
            'model_available': crop_system.model is not None,
            'test_result': result
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'model_available': crop_system.model is not None
        }

@app.post("/crop-recommendation/suitable-crops")
def get_suitable_crops_only(request: CropRecommendationRequest):
    """
    Get suitable crops with probabilities for a location (returns crops with >70% probability)
    
    Input: City/area name (e.g., "Mumbai, India", "Delhi, India")
    Output: List of crops with probabilities [{"crop": "rice", "probability": 0.85}, ...] or []
    
    Example usage:
    - POST /crop-recommendation/suitable-crops
    - Body: {"location": "Mumbai, India", "use_defaults": true}
    - Returns: [{"crop": "rice", "probability": 0.85}]
    """
    try:
        result = crop_system.get_crop_recommendations(
            location=request.location,
            use_defaults=request.use_defaults
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Extract suitable crops with their probabilities (>70% probability)
        suitable_crops = [
            {'crop': rec['crop'], 'probability': rec['confidence']}
            for rec in result['recommendations'] 
            if rec.get('suitability') == 'suitable'
        ]
        
        # Return crop names with probabilities
        return suitable_crops
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting suitable crops: {str(e)}")

@app.post("/crop-recommendation/suitable-crops/manual")
def get_suitable_crops_manual(request: ManualCropRequest):
    """
    Get suitable crops with probabilities using manual environmental data (returns crops with >70% probability)
    
    Input: Environmental parameters (nitrogen, temperature, humidity, ph, rainfall)
    Output: List of crops with probabilities [{"crop": "rice", "probability": 0.85}, ...] or []
    
    Example usage:
    - POST /crop-recommendation/suitable-crops/manual
    - Body: {"nitrogen": 90, "temperature": 23.75, "humidity": 82.32, "ph": 6.5, "rainfall": 200.98}
    - Returns: [{"crop": "rice", "probability": 0.85}]
    """
    try:
        # Prepare model input
        model_input = {
            'N': request.nitrogen,
            'temperature': request.temperature,
            'ph': request.ph,
            'rainfall': request.rainfall,
            'humidity': request.humidity
        }
        
        # Get recommendations from ML model
        recommendations = crop_system._predict_crops(model_input)
        
        # Extract suitable crops with their probabilities (>70% probability)
        suitable_crops = [
            {'crop': rec['crop'], 'probability': rec['confidence']}
            for rec in recommendations 
            if rec.get('suitability') == 'suitable'
        ]
        
        # Return crop names with probabilities
        return suitable_crops
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting suitable crops: {str(e)}")

# User-specific chat memory using session IDs
user_chat_sessions: Dict[str, List[tuple[str, str]]] = {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = request.message
    session_id = request.session_id
    # print(f"Received chat request: {user_input} (Session ID: {session_id})")
    if not user_input:
        raise HTTPException(status_code=400, detail="No message provided.")

    # Generate new session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Initialize session if it doesn't exist
    if session_id not in user_chat_sessions:
        user_chat_sessions[session_id] = []
    
    # Get chat history for this specific session
    chat_turns = user_chat_sessions[session_id]

    # Build formatted prompt with session-specific history
    formatted_past = ""
    for user_msg, assistant_msg in chat_turns:
        formatted_past += f"User: {user_msg}\nAssistant: {assistant_msg}\n"

    full_prompt = (
        "You are a helpful assistant for agricultural advice and recommendations.\n"
        + ("Previous conversation:\n" + formatted_past if formatted_past else "")
        + "\nCurrent question:\n"
        + f"User: {user_input}"
    )

    # Call the agent using formatted message
    result = chat_app.invoke({
        "messages": [{"role": "user", "content": full_prompt}]
    })
    print(f"Session {session_id} - Result: {result['messages']}")
    
    # Extract the final AI response (get the last meaningful AI message)
    ai_response = ""
    
    # Get all AI messages and find the last one with meaningful content
    ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content.strip()]
    
    if ai_messages:
        # Get the last AI message with content
        last_message = ai_messages[-1]
        ai_response = last_message.content
        
        # If the last message is from supervisor and has good content, use it
        # Otherwise, look for the best response from any expert
        if not ai_response or len(ai_response.strip()) < 10:
            # Look for the most substantial response from any AI agent
            for msg in reversed(ai_messages):
                if msg.content.strip() and len(msg.content.strip()) > 20:
                    ai_response = msg.content
                    break
    
    # Fallback if no good response found
    if not ai_response:
        ai_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

    # Save to session-specific chat history
    user_chat_sessions[session_id].append((user_input, ai_response))

    return ChatResponse(reply=ai_response, session_id=session_id)

@app.delete("/chat/clear")
async def clear_chat(session_id: Optional[str] = Query(None)):
    """
    Clear the chat history for a specific session or all sessions
    
    Args:
        session_id: Optional session ID. If provided, clears only that session.
                   If not provided, clears all sessions.
    """
    global user_chat_sessions
    
    if session_id:
        # Clear specific session
        if session_id in user_chat_sessions:
            user_chat_sessions[session_id] = []
            return {
                "success": True,
                "message": f"Chat history cleared for session {session_id}",
                "session_id": session_id,
                "chat_turns_count": 0
            }
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    else:
        # Clear all sessions
        user_chat_sessions = {}
        return {
            "success": True,
            "message": "All chat histories cleared successfully",
            "total_sessions_cleared": len(user_chat_sessions)
        }

@app.get("/chat/history")
async def get_chat_history(session_id: Optional[str] = Query(None)):
    """
    Get the chat history for a specific session or all sessions
    
    Args:
        session_id: Optional session ID. If provided, returns only that session's history.
                   If not provided, returns all sessions.
    """
    if session_id:
        # Get specific session history
        if session_id in user_chat_sessions:
            return {
                "success": True,
                "session_id": session_id,
                "chat_turns": user_chat_sessions[session_id],
                "total_conversations": len(user_chat_sessions[session_id])
            }
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    else:
        # Get all sessions
        return {
            "success": True,
            "all_sessions": user_chat_sessions,
            "total_sessions": len(user_chat_sessions),
            "total_conversations": sum(len(turns) for turns in user_chat_sessions.values())
        }

@app.get("/chat/sessions")
async def get_active_sessions():
    """
    Get list of all active chat sessions
    
    Returns basic information about all active sessions without the full chat history.
    """
    sessions_info = {}
    for session_id, chat_turns in user_chat_sessions.items():
        sessions_info[session_id] = {
            "conversation_count": len(chat_turns),
            "last_activity": chat_turns[-1] if chat_turns else None
        }
    
    return {
        "success": True,
        "active_sessions": sessions_info,
        "total_sessions": len(user_chat_sessions)
    }

