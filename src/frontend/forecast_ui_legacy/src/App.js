import React, { useState, useEffect } from 'react';
import "./App.css";
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Filler,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Filler,
  Title,
  Tooltip,
  Legend
);

const BACKEND_URL = 'http://localhost:8000';

// Connection Test Component
const ConnectionTest = ({ connectionStatus, setConnectionStatus, setError }) => {
  const testConnection = async () => {
    setConnectionStatus('connecting');
    try {
      const response = await fetch(`${BACKEND_URL}/health`);
      if (response.ok) {
        const data = await response.json();
        setConnectionStatus('connected');
        setError(null);
        console.log('✅ Backend connection successful:', data);
        return data;
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (err) {
      console.error('❌ Backend connection failed:', err);
      setConnectionStatus('disconnected');
      setError(`Connection failed: ${err.message}`);
      return null;
    }
  };

  useEffect(() => {
    testConnection();
  }, []);

  return (
    <div className="connection-test card">
      <div className="connection-header">
        <div className="connection-info">
          <h3>🔌 Backend Connection</h3>
          <div className={`status-indicator status-${connectionStatus}`}>
            {connectionStatus === 'connected' ? '✅ Connected' :
             connectionStatus === 'connecting' ? '🔄 Testing...' :
             '❌ Disconnected'}
          </div>
        </div>
        <button
          onClick={testConnection}
          className="btn btn-primary"
        >
          🔄 Test Connection
        </button>
      </div>
    </div>
  );
};

// Daily Forecast Chart Component
const DailyForecastChart = ({ forecastData, granularity }) => {
  if (!forecastData || !forecastData.forecasts || forecastData.forecasts.length === 0) {
    return (
      <div className="chart-container">
        <h3 className="chart-title">📊 Daily Forecast - {granularity}</h3>
        <p className="no-data">No forecast data available</p>
      </div>
    );
  }

  const forecast = forecastData.forecasts[0];
  
  const chartData = {
    labels: forecast.dates.map(date => {
      const d = new Date(date);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }),
    datasets: [
      {
        label: 'Base Forecast',
        data: forecast.base_forecast,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.1)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'Promotional Forecast',
        data: forecast.promotional_forecast,
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'Total Forecast',
        data: forecast.total_forecast,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
        tension: 0.4,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: `Daily Forecast`,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Demand Quantity',
        },
      },
    },
  };

  return (
    <div className="chart-container">
      <div className="chart-wrapper">
        <Line data={chartData} options={chartOptions} />
      </div>
      {forecast.has_historical_data && (
        <div className="historical-stats grid-3">
          <div className="stat-card blue">
            <p className="stat-label">Historical Base Avg</p>
            <p className="stat-value">{forecast.historical_base_avg?.toFixed(1) || 'N/A'}</p>
          </div>
          <div className="stat-card red">
            <p className="stat-label">Historical Promo Avg</p>
            <p className="stat-value">{forecast.historical_promo_avg?.toFixed(1) || 'N/A'}</p>
          </div>
        </div>
      )}
    </div>
  );
};

// Promotional Analysis Chart Component
const PromotionalAnalysisChart = ({ promotionalData }) => {
  if (!promotionalData) {
    return (
      <div className="chart-container">
        <h3 className="chart-title">🎯 Promotional vs Base Demand</h3>
        <p className="no-data">No promotional data available</p>
      </div>
    );
  }

  const chartData = {
    labels: promotionalData.dates.map(date => {
      const d = new Date(date);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }),
    datasets: [
      {
        label: 'Base Demand',
        data: promotionalData.base_demand_forecast,
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
      {
        label: 'Promotional Demand',
        data: promotionalData.promotional_demand_forecast,
        backgroundColor: 'rgba(255, 99, 132, 0.8)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Promotional vs Base Demand Forecast',
      },
    },
    scales: {
      x: {
        stacked: true,
      },
      y: {
        stacked: true,
        beginAtZero: true,
        title: {
          display: true,
          text: 'Demand Quantity',
        },
      },
    },
  };

  return (
    <div className="chart-container">
      <div className="chart-wrapper">
        <Bar data={chartData} options={chartOptions} />
      </div>
      <div className="promotional-summary grid-2">
        <div className="stat-card blue">
          <p className="stat-label">Total Base Demand</p>
          <p className="stat-value">{promotionalData.historical_summary.total_base_demand.toFixed(0)}</p>
        </div>
        <div className="stat-card red">
          <p className="stat-label">Total Promotional Demand</p>
          <p className="stat-value">{promotionalData.historical_summary.total_promotional_demand.toFixed(0)}</p>
        </div>
      </div>
    </div>
  );
};

// Simple Forecast Chart Component
const SimpleForecastChart = ({ data }) => {
  if (!data) {
    return (
      <div className="chart-container">
        <h3 className="chart-title">📈 Forecast Chart</h3>
        <p className="no-data">No forecast data available</p>
      </div>
    );
  }

  const chartData = {
    labels: data.labels,
    datasets: [
      {
        label: 'Forecast',
        data: data.values,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
        tension: 0.4,
      },
      ...(data.confidence_upper ? [{
        label: 'Confidence Upper',
        data: data.confidence_upper,
        borderColor: 'rgba(255, 206, 86, 0.5)',
        backgroundColor: 'transparent',
        fill: false,
        borderDash: [5, 5],
      }] : []),
      ...(data.confidence_lower ? [{
        label: 'Confidence Lower',
        data: data.confidence_lower,
        borderColor: 'rgba(255, 206, 86, 0.5)',
        backgroundColor: 'transparent',
        fill: false,
        borderDash: [5, 5],
      }] : [])
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Supply Chain Demand Forecast',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Demand Quantity',
        },
      },
    },
  };

  return (
    <div className="chart-container">
      <div className="chart-wrapper">
        <Line data={chartData} options={chartOptions} />
      </div>
    </div>
  );
};

// Control Panel Component
const ControlPanel = ({ onDailyForecast, onPromotionalForecast, loading, dataSummary }) => {
  const [activeTab, setActiveTab] = useState('daily');
  const [dailyParams, setDailyParams] = useState({
    sku_id: '',
    channel: '',
    location: '',
    start_date: new Date().toISOString().split('T')[0],
    end_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    forecast_days: 30,
    granularity_level: 'sku_channel_location'
  });
  
  const [promoParams, setPromoParams] = useState({
    sku_id: '',
    channel: '',
    location: '',
    start_date: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end_date: new Date().toISOString().split('T')[0],
    forecast_days: 30,
    include_base_demand: true,
    include_promotional_demand: true,
    promo_type_filter: ''
  });

  return (
    <div className="control-panel card">
      <div className="tab-list">
        <button
          className={`tab-button ${activeTab === 'daily' ? 'active' : ''}`}
          onClick={() => setActiveTab('daily')}
        >
          FR1: Daily Forecast
        </button>
        <button
          className={`tab-button ${activeTab === 'promotional' ? 'active' : ''}`}
          onClick={() => setActiveTab('promotional')}
        >
          FR2: Promotional Analysis
        </button>
      </div>

      {activeTab === 'daily' && (
        <div className="tab-content">
          <h3 className="section-title">📊 Daily Forecast Parameters</h3>
          
          <div className="form-grid grid-4">
            <div className="form-group">
              <label className="form-label">SKU ID</label>
              <input
                type="text"
                value={dailyParams.sku_id}
                onChange={(e) => setDailyParams(prev => ({ ...prev, sku_id: e.target.value }))}
                className="form-input"
                placeholder="Optional"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Channel</label>
              <input
                type="text"
                value={dailyParams.channel}
                onChange={(e) => setDailyParams(prev => ({ ...prev, channel: e.target.value }))}
                className="form-input"
                placeholder="Optional"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Location</label>
              <input
                type="text"
                value={dailyParams.location}
                onChange={(e) => setDailyParams(prev => ({ ...prev, location: e.target.value }))}
                className="form-input"
                placeholder="Optional"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Granularity</label>
              <select
                value={dailyParams.granularity_level}
                onChange={(e) => setDailyParams(prev => ({ ...prev, granularity_level: e.target.value }))}
                className="form-input form-select"
              >
                <option value="sku_channel_location">SKU-Channel-Location</option>
                <option value="sku">SKU Level</option>
                <option value="channel">Channel Level</option>
                <option value="location">Location Level</option>
              </select>
            </div>
          </div>
          
          <div className="form-grid grid-3">
            <div className="form-group">
              <label className="form-label">Start Date</label>
              <input
                type="date"
                value={dailyParams.start_date}
                onChange={(e) => setDailyParams(prev => ({ ...prev, start_date: e.target.value }))}
                className="form-input"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">End Date</label>
              <input
                type="date"
                value={dailyParams.end_date}
                onChange={(e) => setDailyParams(prev => ({ ...prev, end_date: e.target.value }))}
                className="form-input"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Forecast Days</label>
              <input
                type="number"
                value={dailyParams.forecast_days}
                onChange={(e) => setDailyParams(prev => ({ ...prev, forecast_days: parseInt(e.target.value) }))}
                min="1"
                max="365"
                className="form-input"
              />
            </div>
          </div>
          
          <button
            onClick={() => onDailyForecast(dailyParams)}
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? '⏳ Generating...' : '📊 Generate Daily Forecast'}
          </button>
        </div>
      )}

      {activeTab === 'promotional' && (
        <div className="tab-content">
          <h3 className="section-title">🎯 Promotional Analysis Parameters</h3>
          
          <div className="form-grid grid-4">
            <div className="form-group">
              <label className="form-label">SKU ID</label>
              <input
                type="text"
                value={promoParams.sku_id}
                onChange={(e) => setPromoParams(prev => ({ ...prev, sku_id: e.target.value }))}
                className="form-input"
                placeholder="Optional"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Channel</label>
              <input
                type="text"
                value={promoParams.channel}
                onChange={(e) => setPromoParams(prev => ({ ...prev, channel: e.target.value }))}
                className="form-input"
                placeholder="Optional"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Location</label>
              <input
                type="text"
                value={promoParams.location}
                onChange={(e) => setPromoParams(prev => ({ ...prev, location: e.target.value }))}
                className="form-input"
                placeholder="Optional"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Promo Type Filter</label>
              <input
                type="text"
                value={promoParams.promo_type_filter}
                onChange={(e) => setPromoParams(prev => ({ ...prev, promo_type_filter: e.target.value }))}
                className="form-input"
                placeholder="Optional"
              />
            </div>
          </div>
          
          <div className="form-grid grid-3">
            <div className="form-group">
              <label className="form-label">Start Date</label>
              <input
                type="date"
                value={promoParams.start_date}
                onChange={(e) => setPromoParams(prev => ({ ...prev, start_date: e.target.value }))}
                className="form-input"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">End Date</label>
              <input
                type="date"
                value={promoParams.end_date}
                onChange={(e) => setPromoParams(prev => ({ ...prev, end_date: e.target.value }))}
                className="form-input"
              />
            </div>
            
            <div className="form-group">
              <label className="form-label">Forecast Days</label>
              <input
                type="number"
                value={promoParams.forecast_days}
                onChange={(e) => setPromoParams(prev => ({ ...prev, forecast_days: parseInt(e.target.value) }))}
                min="1"
                max="365"
                className="form-input"
              />
            </div>
          </div>
          
          <div className="checkbox-group">
            <div className="checkbox-wrapper">
              <input
                type="checkbox"
                checked={promoParams.include_base_demand}
                onChange={(e) => setPromoParams(prev => ({ ...prev, include_base_demand: e.target.checked }))}
                className="checkbox"
                id="base-demand"
              />
              <label htmlFor="base-demand" className="checkbox-label">Include Base Demand</label>
            </div>
            <div className="checkbox-wrapper">
              <input
                type="checkbox"
                checked={promoParams.include_promotional_demand}
                onChange={(e) => setPromoParams(prev => ({ ...prev, include_promotional_demand: e.target.checked }))}
                className="checkbox"
                id="promo-demand"
              />
              <label htmlFor="promo-demand" className="checkbox-label">Include Promotional Demand</label>
            </div>
          </div>
          
          <button
            onClick={() => onPromotionalForecast(promoParams)}
            disabled={loading}
            className="btn btn-error"
          >
            {loading ? '⏳ Analyzing...' : '🎯 Generate Promotional Analysis'}
          </button>
        </div>
      )}
    </div>
  );
};


// Scenario Panel Component
const ScenarioPanel = ({ onRunScenario, loading }) => {
  const [scenario, setScenario] = useState({
    temperature: 1.0,
    inflation: 1.0,
    discount: 1.0,
    price: 1.0
  });

  const handleSliderChange = (key, value) => {
    setScenario(prev => ({
      ...prev,
      [key]: parseFloat(value)
    }));
  };

  const resetScenario = () => {
    setScenario({
      temperature: 1.0,
      inflation: 1.0,
      discount: 1.0,
      price: 1.0
    });
  };

  return (
    <div className="scenario-panel card">
      <h3 className="section-title">🎯 Scenario Analysis</h3>
      
      <div className="scenario-controls grid-4">
        <div className="slider-container">
          <label className="slider-label">
            🌡️ Temperature Impact: {scenario.temperature.toFixed(2)}x
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={scenario.temperature}
            onChange={(e) => handleSliderChange('temperature', e.target.value)}
            className="slider temperature"
          />
        </div>
        
        <div className="slider-container">
          <label className="slider-label">
            💰 Inflation Impact: {scenario.inflation.toFixed(2)}x
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={scenario.inflation}
            onChange={(e) => handleSliderChange('inflation', e.target.value)}
            className="slider inflation"
          />
        </div>
        
        <div className="slider-container">
          <label className="slider-label">
            🎁 Discount Impact: {scenario.discount.toFixed(2)}x
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={scenario.discount}
            onChange={(e) => handleSliderChange('discount', e.target.value)}
            className="slider discount"
          />
        </div>
        
        <div className="slider-container">
          <label className="slider-label">
            💵 Price Impact: {scenario.price.toFixed(2)}x
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={scenario.price}
            onChange={(e) => handleSliderChange('price', e.target.value)}
            className="slider price"
          />
        </div>
      </div>
      
      <div className="scenario-actions">
        <button
          onClick={() => onRunScenario({ scenario })}
          disabled={loading}
          className="btn btn-primary"
        >
          {loading ? '⏳ Running Scenario...' : '🎯 Run Scenario Analysis'}
        </button>
        <button
          onClick={resetScenario}
          className="btn btn-secondary"
        >
          🔄 Reset
        </button>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [dataSummary, setDataSummary] = useState(null);
  const [dailyForecastData, setDailyForecastData] = useState(null);
  const [promotionalData, setPromotionalData] = useState(null);
  const [simpleForecastData, setSimpleForecastData] = useState(null);
  const [historicalData, setHistoricalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelParams, setModelParams] = useState({
    days: 30,
    startDate: new Date().toISOString().split('T')[0]
  });

  // Load initial data
  const loadDataSummary = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/data-summary`);
      if (!response.ok) {
        throw new Error('Failed to fetch data summary');
      }
      const data = await response.json();
      setDataSummary(data);
      console.log('✅ Data summary loaded:', data);
      setError(null);
    } catch (err) {
      console.error('❌ Error loading data summary:', err);
      setError(`Failed to load data summary: ${err.message}`);
    }
  };

  const loadSimpleForecast = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/forecast?days=${modelParams.days}`);
      if (!response.ok) {
        throw new Error('Failed to get forecast data');
      }
      
      const forecastData = await response.json();
      setSimpleForecastData(forecastData);
      console.log('✅ Simple forecast loaded:', forecastData);
      
      // Generate mock historical data
      const historicalSample = [];
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - 30);
      
      for (let i = 0; i < 30; i++) {
        const date = new Date(startDate);
        date.setDate(startDate.getDate() + i);
        
        historicalSample.push({
          date: date.toISOString().split('T')[0],
          value: Math.max(1000, 2000 + Math.sin(i / 5) * 500 + (Math.random() - 0.5) * 800)
        });
      }
      
      setHistoricalData(historicalSample);
      
    } catch (err) {
      console.error('❌ Error loading simple forecast:', err);
      setError(`Failed to load forecast: ${err.message}`);
    }
  };

  // Initialize data on component mount
  useEffect(() => {
    if (connectionStatus === 'connected') {
      loadDataSummary();
      loadSimpleForecast();
    }
  }, [connectionStatus, modelParams.days]);

  const handleDailyForecast = async (params) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Generating daily forecast with params:', params);
      const response = await fetch(`${BACKEND_URL}/api/daily-forecast`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setDailyForecastData(data);
      console.log('✅ Daily forecast generated:', data);
    } catch (err) {
      console.error('❌ Error generating daily forecast:', err);
      setError(`Daily forecast failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handlePromotionalForecast = async (params) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Generating promotional forecast with params:', params);
      const response = await fetch(`${BACKEND_URL}/api/promotional-forecast`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setPromotionalData(data);
      console.log('✅ Promotional forecast generated:', data);
    } catch (err) {
      console.error('❌ Error generating promotional forecast:', err);
      setError(`Promotional forecast failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleScenarioAnalysis = async ({ scenario }) => {
    setScenarioLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Running scenario analysis with params:', scenario);
      const response = await fetch(`${BACKEND_URL}/api/scenario-forecast`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          forecast_days: modelParams.days,
          start_date: modelParams.startDate,
          scenario: scenario
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      setSimpleForecastData(result);
      console.log('✅ Scenario analysis completed:', result);
    } catch (err) {
      console.error('❌ Scenario forecast failed:', err);
      setError('Scenario forecast failed: ' + err.message);
    } finally {
      setScenarioLoading(false);
    }
  };

  const refreshData = () => {
    if (connectionStatus === 'connected') {
      loadDataSummary();
      loadSimpleForecast();
    }
  };

  const updateModelParams = (newParams) => {
    setModelParams(prev => ({ ...prev, ...newParams }));
  };

  const exportToCSV = () => {
  const exportData = [];
  const timestamp = new Date().toISOString().split('T')[0];
  
  // Export Simple Forecast Data
  if (simpleForecastData) {
    exportData.push({
      filename: `simple_forecast_${timestamp}.csv`,
      content: [
        ['Date', 'Forecasted Value', 'Confidence Upper', 'Confidence Lower'],
        ...simpleForecastData.labels.map((label, index) => [
          label,
          simpleForecastData.values[index],
          simpleForecastData.confidence_upper ? simpleForecastData.confidence_upper[index] : '',
          simpleForecastData.confidence_lower ? simpleForecastData.confidence_lower[index] : ''
        ])
      ].map(row => row.join(',')).join('\n')
    });
  }

  // Export Daily Forecast Data
  if (dailyForecastData && dailyForecastData.forecasts) {
    dailyForecastData.forecasts.forEach((forecast, forecastIndex) => {
      const csvContent = [
        ['Date', 'Base Forecast', 'Promotional Forecast', 'Total Forecast'],
        ...forecast.dates.map((date, index) => [
          date,
          forecast.base_forecast[index] || '',
          forecast.promotional_forecast[index] || '',
          forecast.total_forecast[index] || ''
        ])
      ].map(row => row.join(',')).join('\n');

      // Add metadata as comments at the top
      const metadata = [
        `# Daily Forecast Export - ${timestamp}`,
        `# SKU ID: ${forecast.sku_id || 'All'}`,
        `# Granularity: ${dailyForecastData.granularity_level || 'N/A'}`,
        `# Historical Base Avg: ${forecast.historical_base_avg?.toFixed(2) || 'N/A'}`,
        `# Historical Promo Avg: ${forecast.historical_promo_avg?.toFixed(2) || 'N/A'}`,
        `# Trend Slope: ${forecast.trend_slope?.toFixed(3) || 'N/A'}`,
        `# Has Historical Data: ${forecast.has_historical_data}`,
        ''
      ].join('\n');

      exportData.push({
        filename: `daily_forecast_${forecast.sku_id || 'aggregated'}_${timestamp}.csv`,
        content: metadata + csvContent
      });
    });
  }

  // Export Promotional Analysis Data
  if (promotionalData) {
    const csvContent = [
      ['Date', 'Base Demand Forecast', 'Promotional Demand Forecast'],
      ...promotionalData.dates.map((date, index) => [
        date,
        promotionalData.base_demand_forecast[index] || '',
        promotionalData.promotional_demand_forecast[index] || ''
      ])
    ].map(row => row.join(',')).join('\n');

    // Add promotional summary as metadata
    const metadata = [
      `# Promotional Analysis Export - ${timestamp}`,
      `# Total Base Demand: ${promotionalData.historical_summary?.total_base_demand?.toFixed(0) || 'N/A'}`,
      `# Total Promotional Demand: ${promotionalData.historical_summary?.total_promotional_demand?.toFixed(0) || 'N/A'}`,
      `# Base vs Promo Ratio: ${promotionalData.historical_summary?.promotional_vs_base_ratio?.toFixed(2) || 'N/A'}`,
      ''
    ].join('\n');

    if (promotionalData.promotional_impact_analysis) {
      const impactMetadata = [
        `# Promotional Impact Analysis:`,
        `# Total Promotional Days: ${promotionalData.promotional_impact_analysis.total_promotional_days}`,
        `# Average Promotional Lift: ${promotionalData.promotional_impact_analysis.avg_promotional_lift.toFixed(2)}x`,
        `# Promotional Frequency: ${(promotionalData.promotional_impact_analysis.promotional_frequency * 100).toFixed(1)}%`,
        ''
      ].join('\n');
      
      exportData.push({
        filename: `promotional_analysis_${timestamp}.csv`,
        content: metadata + impactMetadata + csvContent
      });
    } else {
      exportData.push({
        filename: `promotional_analysis_${timestamp}.csv`,
        content: metadata + csvContent
      });
    }
  }

  // Export Historical Data if available
  if (historicalData && historicalData.length > 0) {
    const csvContent = [
      ['Date', 'Historical Value'],
      ...historicalData.map(item => [
        item.date,
        item.value
      ])
    ].map(row => row.join(',')).join('\n');

    exportData.push({
      filename: `historical_data_${timestamp}.csv`,
      content: `# Historical Data Export - ${timestamp}\n\n` + csvContent
    });
  }

  // Create and download all files
  if (exportData.length === 0) {
    alert('No data available to export');
    return;
  }

  // If only one file, download directly
  if (exportData.length === 1) {
    const blob = new Blob([exportData[0].content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = exportData[0].filename;
    link.click();
    window.URL.revokeObjectURL(url);
  } else {
    // Multiple files - download each one
    exportData.forEach((file, index) => {
      setTimeout(() => {
        const blob = new Blob([file.content], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = file.filename;
        link.click();
        window.URL.revokeObjectURL(url);
      }, index * 500); // Stagger downloads by 500ms
    });
  }

  console.log(`✅ Exported ${exportData.length} CSV file(s)`);
};

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-container">
          <div className="header-content">
            <h1 className="app-title">
             Supply Chain Forecast Dashboard
            </h1>
            <div className="header-controls">
              <div className="model-controls">
                <div className="control-group">
                  <label htmlFor="forecast-days" className="control-label">
                    Forecast Days:
                  </label>
                  <input 
                    id="forecast-days"
                    type="number" 
                    value={modelParams.days}
                    onChange={(e) => updateModelParams({ days: parseInt(e.target.value) })}
                    min="1"
                    max="365"
                    className="control-input"
                  />
                </div>
                <div className="control-group">
                  <label htmlFor="start-date" className="control-label">
                   Start Date:
                  </label>
                  <input 
                    id="start-date"
                    type="date" 
                    value={modelParams.startDate}
                    onChange={(e) => updateModelParams({ startDate: e.target.value })}
                    className="control-input"
                  />
                </div>
              </div>
              <button
                onClick={refreshData}
                disabled={connectionStatus !== 'connected'}
                className="btn btn-primary"
              >
                🔄 Refresh
              </button>
              <button
                onClick={exportToCSV}
                disabled={!simpleForecastData}
                className="btn btn-success"
              >
                📊 Export CSV
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="app-main">
        {/* Connection Test */}
        <ConnectionTest 
          connectionStatus={connectionStatus}
          setConnectionStatus={setConnectionStatus}
          setError={setError}
        />

        {error && (
          <div className="alert alert-error">
            <span className="alert-icon">⚠️</span>
            <p className="alert-message">{error}</p>
          </div>
        )}

        {/* Scenario Analysis Panel */}
        <ScenarioPanel
          onRunScenario={handleScenarioAnalysis}
          loading={scenarioLoading}
        />


        {/* Simple Forecast Chart */}
        {simpleForecastData && (
          <section className="main-forecast">
            <h2 className="section-title">📈 Main Forecast Chart</h2>
            <SimpleForecastChart data={simpleForecastData} />
          </section>
        )}

        {/* Advanced Control Panel */}
        <ControlPanel
          onDailyForecast={handleDailyForecast}
          onPromotionalForecast={handlePromotionalForecast}
          loading={loading}
          dataSummary={dataSummary}
        />

        <div className="forecast-results">
          {dailyForecastData && (
            <section className="daily-forecast-section">
              <h2 className="section-title">📊 Daily Forecast Analysis</h2>
              <DailyForecastChart
                forecastData={dailyForecastData}
                granularity={dailyForecastData.granularity_level}
              />
            </section>
          )}

          {promotionalData && (
            <section className="promotional-forecast-section">
              <h2 className="section-title">🎯 Promotional Analysis</h2>
              <PromotionalAnalysisChart promotionalData={promotionalData} />
            </section>
          )}

          {promotionalData?.promotional_impact_analysis && (
            <section className="promotional-impact card">
              <h3 className="section-title">📈 Promotional Impact Analysis</h3>
              <div className="impact-stats grid-4">
                <div className="stat-card blue">
                  <p className="stat-label">🎯 Promotional Days</p>
                  <p className="stat-value">
                    {promotionalData.promotional_impact_analysis.total_promotional_days}
                  </p>
                </div>
                <div className="stat-card green">
                  <p className="stat-label">📈 Average Lift</p>
                  <p className="stat-value">
                    {promotionalData.promotional_impact_analysis.avg_promotional_lift.toFixed(2)}x
                  </p>
                </div>
                <div className="stat-card purple">
                  <p className="stat-label">⏰ Promo Frequency</p>
                  <p className="stat-value">
                    {(promotionalData.promotional_impact_analysis.promotional_frequency * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="stat-card orange">
                  <p className="stat-label">⚖️ Base vs Promo Ratio</p>
                  <p className="stat-value">
                    {promotionalData.historical_summary.promotional_vs_base_ratio.toFixed(2)}
                  </p>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;