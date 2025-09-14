import React, { useState } from 'react';
import './ScenarioPanel.css';

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

  const handleSubmit = (e) => {
    e.preventDefault();
    onRunScenario({ scenario });
  };

  return (
    <div className="scenario-panel card">
      <div className="panel-header">
        <h3 className="panel-title">🎯 Scenario Analysis</h3>
      </div>
      
      <form onSubmit={handleSubmit} className="scenario-form">
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
            type="button"
            onClick={resetScenario}
            className="btn btn-secondary"
          >
            🔄 Reset
          </button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? '⏳ Running Scenario...' : '🎯 Run Scenario Analysis'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ScenarioPanel;
