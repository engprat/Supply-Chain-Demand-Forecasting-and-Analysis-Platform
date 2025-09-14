import React, { useState, useEffect } from 'react';
import './PromotionalAnalysisControl.css';

const PromotionalAnalysisControl = ({ onPromotionalForecast, loading, dataSummary }) => {
  const [params, setParams] = useState({
    sku_id: '',
    channel: '',
    location: '',
    region: 'USA', // New field for regional logic
    start_date: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end_date: new Date().toISOString().split('T')[0],
    forecast_days: 30,
    include_base_demand: true,
    include_promotional_demand: true,
    promo_type_filter: '',
    forecast_granularity: 'daily', // New field for regional preferences
    promo_modeling_approach: 'separated' // New field for regional promo logic
  });

  // Regional configuration
  const regionalConfigs = {
    'China': {
      defaultGranularity: 'monthly',
      defaultPromoApproach: 'embedded',
      forecastDaysOptions: [30, 60, 90],
      description: 'Monthly forecast with embedded promotional effects'
    },
    'Korea': {
      defaultGranularity: 'daily',
      defaultPromoApproach: 'separated',
      forecastDaysOptions: [7, 14, 30, 60],
      description: 'Daily forecast with separate promo/non-promo analysis'
    },
    'USA': {
      defaultGranularity: 'daily',
      defaultPromoApproach: 'separated',
      forecastDaysOptions: [7, 14, 30, 60, 90],
      description: 'Standard daily forecast with promotional separation'
    },
    'Europe': {
      defaultGranularity: 'weekly',
      defaultPromoApproach: 'hybrid',
      forecastDaysOptions: [14, 28, 56, 84],
      description: 'Weekly forecast with hybrid promotional modeling'
    }
  };

  // Update regional settings when region changes
  useEffect(() => {
    const config = regionalConfigs[params.region];
    if (config) {
      setParams(prev => ({
        ...prev,
        forecast_granularity: config.defaultGranularity,
        promo_modeling_approach: config.defaultPromoApproach,
        forecast_days: Math.min(prev.forecast_days, Math.max(...config.forecastDaysOptions))
      }));
    }
  }, [params.region]);

  const handleChange = (param, value) => {
    setParams(prev => ({ ...prev, [param]: value }));
  };

  const handleRegionChange = (region) => {
    setParams(prev => ({ ...prev, region }));
  };

  const currentConfig = regionalConfigs[params.region];

  return (
    <div className="control-panel card">
      <div className="panel-header">
        <h3 className="panel-title">🎯 Promotional Analysis</h3>
        <div className="region-selector">
          <label className="form-label">Region</label>
          <select
            value={params.region}
            onChange={(e) => handleRegionChange(e.target.value)}
            className="form-select region-select"
          >
            {Object.keys(regionalConfigs).map(region => (
              <option key={region} value={region}>{region}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Regional Configuration Info */}
      <div className="regional-info">
        <div className="info-badge">
          <span className="badge-label">Active Configuration:</span>
          <span className="badge-value">{currentConfig.description}</span>
        </div>
      </div>

      <div className="tab-content">
        <div className="form-grid grid-3">
          <div className="form-group">
            <label className="form-label">SKU ID</label>
            <input
              type="text"
              value={params.sku_id}
              onChange={(e) => handleChange('sku_id', e.target.value)}
              className="form-input"
              placeholder="Required"
              required
            />
          </div>
          
          <div className="form-group">
            <label className="form-label">Channel</label>
            <input
              type="text"
              value={params.channel}
              onChange={(e) => handleChange('channel', e.target.value)}
              className="form-input"
              placeholder="Required"
              required
            />
          </div>
          
          <div className="form-group">
            <label className="form-label">Location</label>
            <input
              type="text"
              value={params.location}
              onChange={(e) => handleChange('location', e.target.value)}
              className="form-input"
              placeholder="Required"
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Start Date</label>
            <input
              type="date"
              value={params.start_date}
              onChange={(e) => handleChange('start_date', e.target.value)}
              className="form-input"
              required
            />
          </div>
          
          <div className="form-group">
            <label className="form-label">End Date</label>
            <input
              type="date"
              value={params.end_date}
              onChange={(e) => handleChange('end_date', e.target.value)}
              className="form-input"
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">
              Forecast Days
              <span className="field-hint">({currentConfig.defaultGranularity} granularity)</span>
            </label>
            <select
              value={params.forecast_days}
              onChange={(e) => handleChange('forecast_days', parseInt(e.target.value))}
              className="form-select"
            >
              {currentConfig.forecastDaysOptions.map(days => (
                <option key={days} value={days}>
                  {days} days
                  {currentConfig.defaultGranularity === 'monthly' && ` (~${Math.ceil(days/30)} months)`}
                  {currentConfig.defaultGranularity === 'weekly' && ` (~${Math.ceil(days/7)} weeks)`}
                </option>
              ))}
            </select>
          </div>

          {/* Regional Forecast Granularity */}
          <div className="form-group">
            <label className="form-label">Forecast Granularity</label>
            <select
              value={params.forecast_granularity}
              onChange={(e) => handleChange('forecast_granularity', e.target.value)}
              className="form-select"
              disabled={params.region === 'China'} // China always uses monthly
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
            {params.region === 'China' && (
              <div className="field-note">Monthly granularity required for China region</div>
            )}
          </div>

          {/* Promotional Modeling Approach */}
          <div className="form-group">
            <label className="form-label">Promotional Modeling</label>
            <select
              value={params.promo_modeling_approach}
              onChange={(e) => handleChange('promo_modeling_approach', e.target.value)}
              className="form-select"
            >
              <option value="separated">Separated (Base + Promo)</option>
              <option value="embedded">Embedded (Combined)</option>
              <option value="hybrid">Hybrid (Adaptive)</option>
            </select>
            <div className="field-note">
              {params.promo_modeling_approach === 'separated' && 'Analyzes base and promotional demand separately'}
              {params.promo_modeling_approach === 'embedded' && 'Combines promotional effects into base forecast'}
              {params.promo_modeling_approach === 'hybrid' && 'Uses best approach based on data patterns'}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Promo Type Filter</label>
            <input
              type="text"
              value={params.promo_type_filter}
              onChange={(e) => handleChange('promo_type_filter', e.target.value)}
              className="form-input"
              placeholder="Optional (e.g., seasonal, discount)"
            />
          </div>
        </div>

        {/* Conditional rendering based on promotional modeling approach */}
        {params.promo_modeling_approach === 'separated' && (
          <div className="form-group checkbox-group">
            <label className="form-label">Include in Analysis</label>
            <div className="checkbox-options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={params.include_base_demand}
                  onChange={(e) => handleChange('include_base_demand', e.target.checked)}
                  className="form-checkbox"
                />
                <span>Base Demand</span>
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={params.include_promotional_demand}
                  onChange={(e) => handleChange('include_promotional_demand', e.target.checked)}
                  className="form-checkbox"
                />
                <span>Promotional Demand</span>
              </label>
            </div>
          </div>
        )}

        {params.promo_modeling_approach === 'embedded' && (
          <div className="embedded-note">
            <div className="note-content">
              <strong>Embedded Mode:</strong> Promotional effects are automatically included in the combined forecast.
              No separate base/promotional breakdown will be shown.
            </div>
          </div>
        )}

        <div className="form-actions">
          <button
            onClick={() => onPromotionalForecast(params)}
            className="btn btn-primary"
            disabled={loading || !params.sku_id || !params.channel || !params.location}
          >
            {loading ? 'Analyzing...' : 'Run Promotional Analysis'}
          </button>
          
          {params.region && (
            <div className="action-note">
              Using {params.region} regional configuration: {currentConfig.description}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PromotionalAnalysisControl;