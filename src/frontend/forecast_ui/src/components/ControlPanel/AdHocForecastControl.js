import React, { useState } from 'react';
import './AdHocForecastControl.css';

const AdHocForecastControl = ({ onAdHocAdjustment, loading, dataSummary }) => {
  const [params, setParams] = useState({
    sku_id: '',
    channel: '',
    location: '',
    target_date: new Date().toISOString().split('T')[0],
    adjustment_type: 'absolute',
    adjustment_value: '',
    current_forecast: '',
    reason_category: 'demand_shift',
    notes: '',
    granularity_level: 'sku_channel_location'
  });

  const [previewValue, setPreviewValue] = useState(null);

  const handleChange = (param, value) => {
    setParams(prev => {
      const newParams = { ...prev, [param]: value };
      
      // Auto-calculate preview when adjustment values change
      if ((param === 'adjustment_value' || param === 'current_forecast' || param === 'adjustment_type') && 
          newParams.current_forecast && newParams.adjustment_value) {
        const current = parseFloat(newParams.current_forecast);
        const adjustment = parseFloat(newParams.adjustment_value);
        
        let newValue;
        switch (newParams.adjustment_type) {
          case 'absolute':
            newValue = adjustment;
            break;
          case 'percentage':
            newValue = current * (1 + adjustment / 100);
            break;
          case 'multiplier':
            newValue = current * adjustment;
            break;
          default:
            newValue = current;
        }
        setPreviewValue(Math.round(newValue * 100) / 100);
      } else if (!newParams.current_forecast || !newParams.adjustment_value) {
        setPreviewValue(null);
      }
      
      return newParams;
    });
  };

  const isFormValid = () => {
    return params.sku_id && 
           params.target_date && 
           params.current_forecast && 
           params.adjustment_value && 
           !isNaN(params.current_forecast) && 
           !isNaN(params.adjustment_value);
  };

  return (
    <div className="control-panel card">
      <div className="panel-header">
        <h3 className="panel-title">Ad-Hoc Forecast Adjustments</h3>
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
            <label className="form-label">Target Date</label>
            <input
              type="date"
              value={params.target_date}
              onChange={(e) => handleChange('target_date', e.target.value)}
              className="form-input"
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">Current Forecast</label>
            <input
              type="number"
              value={params.current_forecast}
              onChange={(e) => handleChange('current_forecast', e.target.value)}
              className="form-input"
              placeholder="Enter current value"
              step="0.01"
              required
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
            <label className="form-label">Adjustment Type</label>
            <select
              value={params.adjustment_type}
              onChange={(e) => handleChange('adjustment_type', e.target.value)}
              className="form-input"
            >
              <option value="absolute">Absolute Value</option>
              <option value="percentage">Percentage Change</option>
              <option value="multiplier">Multiplier</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">
              Adjustment Value
              {params.adjustment_type === 'percentage' && ' (%)'}
              {params.adjustment_type === 'multiplier' && ' (x)'}
            </label>
            <input
              type="number"
              value={params.adjustment_value}
              onChange={(e) => handleChange('adjustment_value', e.target.value)}
              className="form-input"
              placeholder={
                params.adjustment_type === 'absolute' ? 'New forecast value' :
                params.adjustment_type === 'percentage' ? 'e.g., 15 for +15%' :
                'e.g., 1.2 for 20% increase'
              }
              step={params.adjustment_type === 'multiplier' ? '0.1' : '1'}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">Reason Category</label>
            <select
              value={params.reason_category}
              onChange={(e) => handleChange('reason_category', e.target.value)}
              className="form-input"
            >
              <option value="demand_shift">Demand Shift</option>
              <option value="market_event">Market Event</option>
              <option value="supply_constraint">Supply Constraint</option>
              <option value="promotional_impact">Promotional Impact</option>
              <option value="seasonal_adjustment">Seasonal Adjustment</option>
              <option value="external_factor">External Factor</option>
              <option value="data_correction">Data Correction</option>
              <option value="other">Other</option>
            </select>
          </div>
        </div>

        {previewValue !== null && (
          <div className="adjustment-preview">
            <h4>Adjustment Preview:</h4>
            <div className="preview-row">
              <span className="current-value">Current: {params.current_forecast}</span>
              <span className="arrow">→</span>
              <span className="new-value">New: {previewValue}</span>
              <span className="change-value">
                ({previewValue > params.current_forecast ? '+' : ''}
                {Math.round((previewValue - params.current_forecast) * 100) / 100})
              </span>
            </div>
          </div>
        )}

        <div className="form-group full-width">
          <label className="form-label">Notes</label>
          <textarea
            value={params.notes}
            onChange={(e) => handleChange('notes', e.target.value)}
            className="form-input"
            placeholder="Optional: Provide additional context for this adjustment..."
            rows="3"
          />
        </div>

        <div className="form-actions">
          <button
            onClick={() => onAdHocAdjustment(params)}
            className="btn btn-primary"
            disabled={loading || !isFormValid()}
          >
            {loading ? 'Applying...' : 'Apply Adjustment Immediately'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AdHocForecastControl;