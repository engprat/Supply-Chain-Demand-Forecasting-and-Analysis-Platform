import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './ScenarioPanel.css';

const DEFAULT_SCENARIO = {
  order_qty_units: 1000, // units
  revenue_dollars: 100000, // $
  temperature_c: 20, // °C
  price_dollars: 100, // $
  discount_percent: 10, // %
  promotion_percent: 0, // %
  inflation_percent: 2, // %
};

const ScenarioPanel = ({ onRunScenario, loading }) => {
  const [scenario, setScenario] = useState(DEFAULT_SCENARIO);
  
  const [error, setError] = useState(null);

  // Convert real-world units to multipliers for scenario logic
  const getScenarioWithMultiplier = (scen) => ({
    order_qty: scen.order_qty_units / 1000, // 1000 units = 1.0
    revenue: scen.revenue_dollars / 100000, // $100,000 = 1.0
    temperature: 1 + (scen.temperature_c - 20) * 0.01, // 20°C = 1.0
    price: scen.price_dollars / 100, // $100 = 1.0
    discount: 1 - scen.discount_percent / 100, // 10% = 0.9 multiplier
    promotion: 1 + scen.promotion_percent / 100, // 0% = 1.0, 100% = 2.0
    inflation: 1 + scen.inflation_percent / 100, // 2% = 1.02
  });

  const handleChange = (key, value) => {
    setScenario((prev) => ({ ...prev, [key]: isNaN(value) ? 0 : value }));
  };

  const handleRunScenario = async (e) => {
    e.preventDefault();
    setError(null);
    
    try {
      await onRunScenario({ scenario: getScenarioWithMultiplier(scenario) });
    } catch (err) {
      setError('Scenario forecast failed: ' + err.message);
    }
  };


  return (
    <div className="scenario-panel">
      <h3>🔎 Scenario Simulation</h3>
      <form onSubmit={handleRunScenario} className="scenario-form">
        <div className="scenario-fields">
          <label>
            Order Quantity
            <input
              type="range"
              min="0"
              max="10000"
              step="10"
              value={scenario.order_qty_units}
              onChange={e => handleChange('order_qty_units', parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="slider-value">{scenario.order_qty_units} units</span>
          </label>
          <label>
            Revenue
            <input
              type="range"
              min="0"
              max="1000000"
              step="1000"
              value={scenario.revenue_dollars}
              onChange={e => handleChange('revenue_dollars', parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="slider-value">${(scenario.revenue_dollars ?? 0).toLocaleString()}</span>
          </label>
          <label>
            Temperature (°C)
            <input
              type="range"
              min="-20"
              max="50"
              step="1"
              value={scenario.temperature_c}
              onChange={e => handleChange('temperature_c', parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="slider-value">{scenario.temperature_c}&deg;C</span>
          </label>
          <label>
            Price
            <input
              type="range"
              min="0"
              max="500"
              step="1"
              value={scenario.price_dollars}
              onChange={e => handleChange('price_dollars', parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="slider-value">${scenario.price_dollars}</span>
          </label>
          <label>
            Discount
            <input
              type="range"
              min="0"
              max="80"
              step="1"
              value={scenario.discount_percent}
              onChange={e => handleChange('discount_percent', parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="slider-value">{scenario.discount_percent}%</span>
          </label>
          <label>
            Promotion
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={scenario.promotion_percent}
              onChange={e => handleChange('promotion_percent', parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="slider-value">{scenario.promotion_percent}%</span>
          </label>
          <label>
            Inflation
            <input
              type="range"
              min="-10"
              max="20"
              step="1"
              value={scenario.inflation_percent}
              onChange={e => handleChange('inflation_percent', parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="slider-value">{scenario.inflation_percent}%</span>
          </label>
          
        </div>
        <button type="submit" className="run-scenario-btn" disabled={loading} aria-busy={loading}>
          {loading ? 'Running...' : 'Run Scenario'}
        </button>
        {error && <div className="scenario-error">{error}</div>}
      </form>
      <div className="scenario-tip">Tip: Set multipliers (e.g., 1.1 for +10%) to simulate changes.</div>
    </div>
  );
};

ScenarioPanel.propTypes = {
  onRunScenario: PropTypes.func.isRequired,
  loading: PropTypes.bool
};

export default ScenarioPanel;
