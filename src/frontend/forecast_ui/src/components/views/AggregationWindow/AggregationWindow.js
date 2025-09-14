import React, { useState } from 'react';
import AggregationControl from '../../ControlPanel/AggregationControl';
import './AggregationWindow.css';

// Configuration for backend URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const AggregationWindow = () => {
  const [loading, setLoading] = useState(false);
  const [aggregationData, setAggregationData] = useState(null);
  const [drillData, setDrillData] = useState(null);
  const [error, setError] = useState(null);

  const handleAggregate = async (params) => {
    console.log('AggregationWindow: Received params:', params);
    setLoading(true);
    setError(null);
    setDrillData(null); // Clear drill data when new aggregation is requested
    
    try {
      const url = `${API_BASE_URL}/api/aggregate-forecast`;
      console.log('Making request to:', url);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });

      console.log('Response status:', response.status);
      const responseText = await response.text();
      console.log('Raw response:', responseText);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} - ${responseText}`);
      }

      const data = JSON.parse(responseText);
      console.log('Parsed response data:', data);
      setAggregationData(data);
      
    } catch (error) {
      console.error('Aggregation error:', error);
      setError(`Failed to generate aggregated forecast: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDrillDown = async (drillRequest) => {
    console.log('AggregationWindow: Received drill request:', drillRequest);
    setLoading(true);
    setError(null);
    
    try {
      const url = `${API_BASE_URL}/api/drill-down`;
      console.log('Making drill request to:', url);
      console.log('Drill request payload:', JSON.stringify(drillRequest, null, 2));
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(drillRequest),
      });

      console.log('Drill response status:', response.status);
      const responseText = await response.text();
      console.log('Drill response:', responseText);

      if (!response.ok) {
        // Try to parse error response for better error message
        let errorMessage;
        try {
          const errorData = JSON.parse(responseText);
          errorMessage = errorData.detail || errorData.message || responseText;
        } catch {
          errorMessage = responseText;
        }
        throw new Error(`HTTP error! status: ${response.status} - ${errorMessage}`);
      }

      const data = JSON.parse(responseText);
      console.log('Parsed drill data:', data);
      setDrillData(data);
      setAggregationData(null); // Clear single aggregation when drilling
      
    } catch (error) {
      console.error('Drill-down error:', error);
      setError(`Failed to drill down: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to format numbers
  const formatNumber = (num) => {
    if (num === null || num === undefined || isNaN(num)) return '0';
    return Number(num).toLocaleString();
  };

  // Helper function to render forecast values with better display
  const renderForecastPreview = (forecastData) => {
    if (!forecastData || !forecastData.forecast_values) return null;

    const values = forecastData.forecast_values;
    const dates = forecastData.dates || [];
    
    return (
      <div className="forecast-preview">
        <h4>Forecast Preview (First 7 Days):</h4>
        <div className="forecast-values">
          {values.slice(0, 7).map((value, index) => (
            <span key={index} className="forecast-value">
              {dates[index] ? `${dates[index].slice(5)}: ` : `Day ${index + 1}: `}
              {formatNumber(value)}
            </span>
          ))}
        </div>
        {values.every(v => v === 0) && (
          <div className="forecast-note" style={{ marginTop: '10px', color: '#666', fontSize: '14px' }}>
            Note: Forecast values are 0 - this may indicate data filtering issues or missing base demand data.
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="aggregation-window">
      <AggregationControl
        onAggregate={handleAggregate}
        onDrillDown={handleDrillDown}
        loading={loading}
      />

      {error && (
        <div className="error-message">
          <div className="error-content">
            <h4>⚠️ Error</h4>
            <p>{error}</p>
            <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
              <p>Debug info:</p>
              <p>API Base URL: {API_BASE_URL}</p>
              <p>Check browser console for detailed error logs</p>
              <p>Check that your backend is running on the correct port</p>
            </div>
            <button 
              onClick={() => setError(null)}
              style={{ marginTop: '10px', padding: '5px 10px', fontSize: '12px' }}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {(aggregationData || drillData) && (
        <section className="card">
          <h2 className="section-title">📊 Multi-Level Forecast Analysis</h2>
          <div className="results-container">
            {aggregationData && (
              <div className="aggregation-results">
                <h3>Aggregation Results - {aggregationData.level}</h3>
                
                <div className="results-grid">
                  <div className="result-item">
                    <strong>Level:</strong> {aggregationData.level}
                  </div>
                  <div className="result-item">
                    <strong>Records:</strong> {formatNumber(aggregationData.summary_metrics?.total_records || 0)}
                  </div>
                  <div className="result-item">
                    <strong>Quality Score:</strong> {aggregationData.data_quality?.overall_score || 0}%
                  </div>
                  <div className="result-item">
                    <strong>Forecast Days:</strong> {aggregationData.forecast_data?.forecast_values?.length || 0}
                  </div>
                  {aggregationData.summary_metrics?.demand_metrics && (
                    <>
                      <div className="result-item">
                        <strong>Total Demand:</strong> {formatNumber(aggregationData.summary_metrics.demand_metrics.total_demand)}
                      </div>
                      <div className="result-item">
                        <strong>Avg Daily Demand:</strong> {formatNumber(aggregationData.summary_metrics.demand_metrics.average_daily_demand)}
                      </div>
                    </>
                  )}
                  {aggregationData.summary_metrics?.date_range && (
                    <div className="result-item">
                      <strong>Date Range:</strong> {aggregationData.summary_metrics.date_range.start_date} to {aggregationData.summary_metrics.date_range.end_date}
                    </div>
                  )}
                </div>

                {renderForecastPreview(aggregationData.forecast_data)}

                {/* Drill Options Display */}
                {(aggregationData.drill_down_options?.length > 0 || aggregationData.roll_up_options?.length > 0) && (
                  <div className="drill-options-display" style={{ marginTop: '20px' }}>
                    <h4>Available Navigation:</h4>
                    {aggregationData.drill_down_options?.length > 0 && (
                      <p><strong>Drill Down to:</strong> {aggregationData.drill_down_options.map(opt => opt.display_name).join(', ')}</p>
                    )}
                    {aggregationData.roll_up_options?.length > 0 && (
                      <p><strong>Roll Up to:</strong> {aggregationData.roll_up_options.map(opt => opt.display_name).join(', ')}</p>
                    )}
                  </div>
                )}

                {/* Debug Info in Development */}
                {process.env.NODE_ENV === 'development' && (
                  <details style={{ marginTop: '20px', fontSize: '12px' }}>
                    <summary>Debug: Raw Aggregation Data</summary>
                    <pre style={{ background: '#f5f5f5', padding: '10px', overflow: 'auto', maxHeight: '200px' }}>
                      {JSON.stringify(aggregationData, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
            
            {drillData && (
              <div className="drill-results">
                <h3>Drill Results - {drillData.target_level}</h3>
                <div className="drill-summary">
                  Found {drillData.drill_results?.length || 0} drill results
                </div>
                
                {drillData.drill_results?.map((result, index) => (
                  <div key={index} className="drill-item">
                    <strong>{result.aggregation_key}</strong>
                    <span>Records: {formatNumber(result.summary_metrics?.total_records || 0)}</span>
                    {result.summary_metrics?.demand_metrics && (
                      <span>Total Demand: {formatNumber(result.summary_metrics.demand_metrics.total_demand)}</span>
                    )}
                  </div>
                ))}

                {/* Debug Info for Drill Data */}
                {process.env.NODE_ENV === 'development' && (
                  <details style={{ marginTop: '20px', fontSize: '12px' }}>
                    <summary>Debug: Raw Drill Data</summary>
                    <pre style={{ background: '#f5f5f5', padding: '10px', overflow: 'auto', maxHeight: '200px' }}>
                      {JSON.stringify(drillData, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>
        </section>
      )}

      {!loading && !aggregationData && !drillData && !error && (
        <section className="card">
          <div className="welcome-section">
            <h2 className="section-title">🎯 Multi-Level Aggregation</h2>
            <div className="feature-grid">
              <div className="feature-card">
                <h3>📈 Roll Up</h3>
                <p>Aggregate from SKU level up to National level for strategic planning.</p>
              </div>
              <div className="feature-card">
                <h3>🔍 Drill Down</h3>
                <p>Navigate from high-level summaries down to granular SKU-Location-Customer details.</p>
              </div>
              <div className="feature-card">
                <h3>🎛️ Multi-Dimensional</h3>
                <p>Filter and analyze across multiple dimensions: geography, channels, customers, and time.</p>
              </div>
              <div className="feature-card">
                <h3>📊 Real-Time</h3>
                <p>Dynamic forecasting powered by your processed dataset with instant aggregation.</p>
              </div>
            </div>
            <div className="getting-started">
              <h3>Getting Started</h3>
              <ol>
                <li>Select your target aggregation level (National, SKU, SKU-Location, etc.)</li>
                <li>Apply filters for specific dimensions (optional)</li>
                <li>Click "Generate Aggregated Forecast"</li>
                <li>Use drill-down buttons to explore more granular data</li>
              </ol>
              <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#e8f5e8', borderRadius: '5px' }}>
                <strong>💡 Tip:</strong> Start with "National Level" to see all your data aggregated, then drill down to more specific levels.
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default AggregationWindow;