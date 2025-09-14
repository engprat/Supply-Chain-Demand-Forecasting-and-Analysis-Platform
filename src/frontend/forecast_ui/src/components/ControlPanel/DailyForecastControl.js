import React, { useState } from 'react';
import './DailyForecastControl.css';

const DailyForecastControl = ({ onDailyForecast, loading, dataSummary }) => {
  const [params, setParams] = useState({
    sku_id: '',
    channel: '',
    location: '',
    start_date: new Date().toISOString().split('T')[0],
    end_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    forecast_days: 30,
    granularity_level: 'sku_channel_location'
  });

  const handleChange = (param, value) => {
    setParams(prev => ({ ...prev, [param]: value }));
  };

  return (
    <div className="control-panel card">
      <div className="panel-header">
        <h3 className="panel-title">Daily Forecast</h3>
      </div>
      <div className="tab-content">
        <div className="form-grid grid-4">
          <div className="form-group">
            <label className="form-label">SKU ID</label>
            <input
              type="text"
              value={params.sku_id}
              onChange={(e) => handleChange('sku_id', e.target.value)}
              className="form-input"
              placeholder="Optional"
            />
          </div>
          <div className="form-group">
            <label className="form-label">Channel</label>
            <input
              type="text"
              value={params.channel}
              onChange={(e) => handleChange('channel', e.target.value)}
              className="form-input"
              placeholder="Optional"
            />
          </div>
          <div className="form-group">
            <label className="form-label">Location</label>
            <input
              type="text"
              value={params.location}
              onChange={(e) => handleChange('location', e.target.value)}
              className="form-input"
              placeholder="Optional"
            />
          </div>
          <div className="form-group">
            <label className="form-label">Granularity</label>
            <select
              value={params.granularity_level}
              onChange={(e) => handleChange('granularity_level', e.target.value)}
              className="form-input"
            >
              <option value="sku_channel_location">SKU + Channel + Location</option>
              <option value="sku_channel">SKU + Channel</option>
              <option value="sku">SKU Only</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Start Date</label>
            <input
              type="date"
              value={params.start_date}
              onChange={(e) => handleChange('start_date', e.target.value)}
              className="form-input"
            />
          </div>
          <div className="form-group">
            <label className="form-label">End Date</label>
            <input
              type="date"
              value={params.end_date}
              onChange={(e) => handleChange('end_date', e.target.value)}
              className="form-input"
            />
          </div>
          <div className="form-group">
            <label className="form-label">Forecast Days</label>
            <input
              type="number"
              value={params.forecast_days}
              onChange={(e) => handleChange('forecast_days', parseInt(e.target.value) || 30)}
              className="form-input"
              min="1"
              max="365"
            />
          </div>
        </div>
        <div className="form-actions">
          <button
            onClick={() => onDailyForecast(params)}
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Generate Daily Forecast'}
          </button>
        </div>
      </div>
      
    </div>
  );
};

export default DailyForecastControl;
