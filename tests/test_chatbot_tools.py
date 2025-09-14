"""
Integration tests for the chatbot tools.

These tests verify that the tool implementations work correctly with the API endpoints.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
from aiolimiter import AsyncLimiter
import pytest
import pytest_asyncio
import pytest_httpx

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chatbot.tools import (
    GetForecastTool,
    GetForecastVsActualTool,
    GetForecastByCategoryTool,
    GetDemandHeatmapTool,
    GetForecastAccuracyTool,
    ToolExecutionError,
    ToolRateLimitError
)

# Test configuration
TEST_API_URL = "http://testserver:8000"
os.environ["FORECAST_API_URL"] = TEST_API_URL

# Test data
def get_sample_forecast() -> Dict[str, Any]:
    """Generate sample forecast data for testing."""
    return {
        "forecast": [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "value": 100 + i * 10,
                "lower_bound": 80 + i * 10,
                "upper_bound": 120 + i * 10
            }
            for i in range(1, 31)
        ],
        "metadata": {
            "model": "test_model",
            "generated_at": datetime.utcnow().isoformat()
        }
    }

# Fixtures
@pytest_asyncio.fixture
async def forecast_tool():
    """Fixture for GetForecastTool instance."""
    tool = GetForecastTool()
    tool.rate_limiter = AsyncLimiter(10, 60)
    return tool

@pytest_asyncio.fixture
async def forecast_vs_actual_tool():
    """Fixture for GetForecastVsActualTool instance."""
    tool = GetForecastVsActualTool()
    tool.rate_limiter = AsyncLimiter(10, 60)
    return tool

@pytest_asyncio.fixture
async def forecast_by_category_tool():
    """Fixture for GetForecastByCategoryTool instance."""
    tool = GetForecastByCategoryTool()
    tool.rate_limiter = AsyncLimiter(10, 60)
    return tool

@pytest_asyncio.fixture
async def demand_heatmap_tool():
    """Fixture for GetDemandHeatmapTool instance."""
    tool = GetDemandHeatmapTool()
    tool.rate_limiter = AsyncLimiter(10, 60)
    return tool

@pytest_asyncio.fixture
async def forecast_accuracy_tool():
    """Fixture for GetForecastAccuracyTool instance."""
    tool = GetForecastAccuracyTool()
    tool.rate_limiter = AsyncLimiter(10, 60)
    return tool

@pytest.fixture
def mock_forecast_response():
    """Fixture for mock forecast response."""
    return get_sample_forecast()

# Test cases
class TestForecastTools:
    """Test cases for the forecast-related tools."""
    
    @pytest.mark.asyncio
    async def test_get_forecast_tool_success(
        self, 
        forecast_tool: GetForecastTool,
        httpx_mock: pytest_httpx.HTTPXMock,
        mock_forecast_response: Dict[str, Any]
    ):
        """Test successful execution of GetForecastTool."""
        # Mock the API response
        httpx_mock.add_response(
            url=f"{TEST_API_URL}/api/forecast/enhanced?days=30&confidence_level=0.95",
            json=mock_forecast_response,
            status_code=200
        )
        
        # Test with default parameters
        result = await forecast_tool.execute({})
        assert result["success"] is True
        assert "data" in result
        assert "metadata" in result
        assert len(result["data"]["forecast"]) > 0
        
        # Verify the request was made correctly
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request = requests[0]
        assert request is not None
        assert "days=30" in str(request.url.query)
        assert "confidence_level=0.95" in str(request.url.query)
    
    @pytest.mark.asyncio
    async def test_get_forecast_vs_actual_tool_success(
        self, 
        forecast_vs_actual_tool: GetForecastVsActualTool,
        httpx_mock: pytest_httpx.HTTPXMock
    ):
        """Test successful execution of GetForecastVsActualTool."""
        # Mock the API response
        httpx_mock.add_response(
            url=f"{TEST_API_URL}/api/forecast/vs-actual?historical_days=60&forecast_days=30&confidence_level=0.95&start_date=",
            json={
                "historical": [
                    {"date": "2025-01-01", "actual": 100, "forecast": None},
                    {"date": "2025-01-02", "actual": 110, "forecast": None},
                    {"date": "2025-01-03", "actual": 105, "forecast": None}
                ],
                "forecast": [
                    {"date": "2025-01-04", "actual": None, "forecast": 108},
                    {"date": "2025-01-05", "actual": None, "forecast": 112},
                    {"date": "2025-01-06", "actual": None, "forecast": 115}
                ]
            },
            status_code=200
        )
        
        # Test with default parameters
        result = await forecast_vs_actual_tool.execute({})
        assert result["success"] is True
        assert "data" in result
        assert "historical" in result["data"]
        assert "forecast" in result["data"]
        
        # Verify the request was made correctly
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request = requests[0]
        assert request is not None
    
    @pytest.mark.asyncio
    async def test_get_forecast_by_category_tool_success(
        self, 
        forecast_by_category_tool: GetForecastByCategoryTool,
        httpx_mock: pytest_httpx.HTTPXMock
    ):
        """Test successful execution of GetForecastByCategoryTool."""
        # Mock the API response
        httpx_mock.add_response(
            url=f"{TEST_API_URL}/api/forecast/by-category?category_type=product&top_n=5&forecast_days=30&min_confidence=0.7&start_date=",
            json={
                "categories": ["Electronics", "Clothing", "Groceries"],
                "forecasts": {
                    "Electronics": [100, 110, 120],
                    "Clothing": [80, 85, 90],
                    "Groceries": [200, 210, 220]
                },
                "dates": ["2025-01-01", "2025-01-02", "2025-01-03"]
            },
            status_code=200
        )
        
        # Test with parameters
        result = await forecast_by_category_tool.execute({
            "category_type": "product",
            "top_n": 5
        })
        
        assert result["success"] is True
        assert "data" in result
        assert "categories" in result["data"]
        
        # Verify the request was made correctly
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request = requests[0]
        assert request is not None
    
    @pytest.mark.asyncio
    async def test_get_demand_heatmap_tool_success(
        self, 
        demand_heatmap_tool: GetDemandHeatmapTool,
        httpx_mock: pytest_httpx.HTTPXMock
    ):
        """Test successful execution of GetDemandHeatmapTool."""
        # Mock the API response
        httpx_mock.add_response(
            url=f"{TEST_API_URL}/api/forecast/heatmap?x_dimension=day_of_week&y_dimension=category&days_back=30&forecast_days=30&aggregation=sum&normalize=false&fill_na=0",
            json={
                "data": [[10, 20, 30], [20, 30, 40], [30, 40, 50]],
                "x_labels": ["Mon", "Wed", "Fri"],
                "y_labels": ["Electronics", "Clothing", "Groceries"]
            },
            status_code=200
        )
        
        # Test with parameters
        result = await demand_heatmap_tool.execute({
            "x_dimension": "day_of_week",
            "y_dimension": "category",
            "days_back": 30
        })
        
        assert result["success"] is True
        assert "data" in result
        assert "x_labels" in result["data"]
        
        # Verify the request was made correctly
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request = requests[0]
        assert request is not None
    
    @pytest.mark.asyncio
    async def test_get_forecast_accuracy_tool_success(
        self, 
        forecast_accuracy_tool: GetForecastAccuracyTool,
        httpx_mock: pytest_httpx.HTTPXMock
    ):
        """Test successful execution of GetForecastAccuracyTool."""
        # Mock the API response
        mock_response = {
            "metrics": {
                "mape": 12.5,
                "mae": 15.3,
                "rmse": 20.1
            },
            "by_category": {
                "Electronics": {"mape": 10.2, "mae": 12.5, "rmse": 18.3},
                "Clothing": {"mape": 14.8, "mae": 18.1, "rmse": 22.7}
            },
            "forecast_days": 30,
            "historical_days": 90
        }
        
        httpx_mock.add_response(
            url=f"{TEST_API_URL}/api/forecast/accuracy?metric=mape&days_back=90&include_confidence=true&min_samples=5&additional_metrics=rmse%2Cmape&group_by=product",
            json=mock_response,
            status_code=200
        )
        
        # Test with default parameters
        result = await forecast_accuracy_tool.execute({})
        
        assert result["success"] is True
        assert "data" in result
        assert "metrics" in result["data"]
        
        # Verify the request was made correctly
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request = requests[0]
        assert request is not None
    
    @pytest.mark.asyncio
    async def test_tool_rate_limiting(
        self, 
        forecast_tool: GetForecastTool,
        httpx_mock: pytest_httpx.HTTPXMock
    ):
        """Test that rate limiting is handled properly."""
        # Mock a rate limit response for all attempts
        for _ in range(4):
            httpx_mock.add_response(
                status_code=429,
                headers={"Retry-After": "5"}
            )
        
        # Test that the rate limit error is properly raised
        with pytest.raises(ToolRateLimitError) as exc_info:
            await forecast_tool.execute({})
            
        assert "API rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.retry_after == 5.0
    
    @pytest.mark.asyncio
    async def test_tool_http_error(
        self, 
        forecast_tool: GetForecastTool,
        httpx_mock: pytest_httpx.HTTPXMock
    ):
        """Test handling of HTTP errors."""
        # Mock an error response for all 4 attempts (1 initial + 3 retries)
        for _ in range(4):
            httpx_mock.add_response(
                status_code=500,
                text="Internal Server Error"
            )
        
        # Test that the error is properly raised
        with pytest.raises(ToolExecutionError) as exc_info:
            await forecast_tool.execute({})
            
        assert "API request failed with status 500" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_tool_network_error(
        self, 
        forecast_tool: GetForecastTool,
        httpx_mock: pytest_httpx.HTTPXMock
    ):
        """Test handling of network errors."""
        # Mock a network error for all 4 attempts
        for _ in range(4):
            httpx_mock.add_exception(httpx.RequestError("Network error"))
        
        # Test that the error is properly raised
        with pytest.raises(ToolExecutionError) as exc_info:
            await forecast_tool.execute({})
            
        assert "Request failed" in str(exc_info.value)

# Run the tests
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", "-s", __file__]))
