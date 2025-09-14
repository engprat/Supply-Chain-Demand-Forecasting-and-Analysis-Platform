import os
import json
import httpx
import pytest
from datetime import datetime
from typing import Dict, Any

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Sample forecast data for testing
def get_sample_forecast_data() -> Dict[str, Any]:
    return {
        "forecast_days": 30,
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "forecast_values": [100, 120, 110, 130, 140],
        "historical_data": {
            "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "values": [90, 100, 110]
        }
    }

def test_chatbot_endpoint():
    """Test the chatbot endpoint with various scenarios."""
    
    # Test 1: Basic chat without context
    response = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": "Hello KayCee!",
            "context": None,
            "temperature": 0.7,
            "max_output_tokens": 1000
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] != ""
    assert "timestamp" in data
    assert "context_summary" in data
    
    # Test 2: Chat with forecast context
    forecast_data = get_sample_forecast_data()
    response = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": "What's the forecast for tomorrow?",
            "context": {
                "forecast_data": forecast_data,
                "historical_data": forecast_data["historical_data"]
            },
            "temperature": 0.7,
            "max_output_tokens": 1000
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] != ""
    assert "timestamp" in data
    assert "context_summary" in data
    
    # Test 3: Chat with multiple messages in context
    previous_response = data["response"]
    response = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": "How does this compare to last year?",
            "context": {
                "forecast_data": forecast_data,
                "historical_data": forecast_data["historical_data"],
                "api_responses": {
                    "forecast-vs-actual": [{
                        "annotations": [
                            {"date": "2024-01-01", "label": "Trend Up"}
                        ]
                    }]
                }
            },
            "temperature": 0.7,
            "max_output_tokens": 1000
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] != ""
    assert "timestamp" in data
    assert "context_summary" in data
    
    # Test 4: Error handling - invalid message
    response = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": "",  # Empty message
            "context": None,
            "temperature": 0.7,
            "max_output_tokens": 1000
        }
    )
    assert response.status_code == 422  # Should return validation error
    
    # Test 5: Error handling - invalid temperature
    response = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": "Hello",
            "context": None,
            "temperature": 2.0,  # Invalid temperature
            "max_output_tokens": 1000
        }
    )
    assert response.status_code == 422  # Should return validation error

def test_chatbot_error_handling():
    """Test error handling scenarios."""
    
    # Test 1: Missing API key
    os.environ.pop('GEMINI_API_KEY', None)
    response = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": "Hello",
            "context": None,
            "temperature": 0.7,
            "max_output_tokens": 1000
        }
    )
    assert response.status_code == 401
    assert "Gemini API key not configured" in response.json()["detail"]
    
    # Restore API key for other tests
    os.environ['GEMINI_API_KEY'] = "test_api_key"
    
    # Test 2: Invalid API key
    response = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": "Hello",
            "context": None,
            "temperature": 0.7,
            "max_output_tokens": 1000
        }
    )
    assert response.status_code == 500
    assert "Failed to generate response" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main([__file__])
