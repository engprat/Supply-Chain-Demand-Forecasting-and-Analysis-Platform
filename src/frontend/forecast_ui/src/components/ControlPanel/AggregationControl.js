import React, { useState, useEffect } from 'react';
import './AggregationControl.css';

// Configuration for backend URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const AggregationControl = ({ onAggregate, onDrillDown, loading, dataSummary }) => {
  const [params, setParams] = useState({
    level: 'brand_day',
    forecast_days: 30,
    brand: '',
    location: '',
    channel: '',
    customer: ''
  });

  const [dimensionValues, setDimensionValues] = useState({
    brands: [],
    locations: [],
    channels: [],
    customers: []
  });

  const [currentLevel, setCurrentLevel] = useState('brand_day');
  const [drillOptions, setDrillOptions] = useState({ drill_down: [], drill_up: [] });

  const aggregationLevels = [
    { value: 'national_day', label: 'National Level (All Data)' },
    { value: 'brand_day', label: 'Brand Level' },
    { value: 'brand_location_day', label: 'Brand-Location Level' },
    { value: 'brand_channel_day', label: 'Brand-Channel Level' },
    { value: 'sku_day', label: 'SKU Level (Most Detailed)' }
  ];

  // Load dimension values on mount
  useEffect(() => {
    const loadDimensionValues = async () => {
      try {
        // Request more items by increasing the limit
        const [brands, locations, channels, customers] = await Promise.all([
          fetch(`${API_BASE_URL}/api/dimension-values/brand?limit=100`).then(r => r.json()).catch(() => ({ values: [] })),
          fetch(`${API_BASE_URL}/api/dimension-values/location?limit=100`).then(r => r.json()).catch(() => ({ values: [] })),
          fetch(`${API_BASE_URL}/api/dimension-values/channel?limit=100`).then(r => r.json()).catch(() => ({ values: [] })),
          fetch(`${API_BASE_URL}/api/dimension-values/customer?limit=100`).then(r => r.json()).catch(() => ({ values: [] }))
        ]);

        console.log('Dimension values loaded:', {
          brands: brands.values?.length || 0,
          locations: locations.values?.length || 0,
          channels: channels.values?.length || 0,
          customers: customers.values?.length || 0
        });

        setDimensionValues({
          brands: brands.values || [],
          locations: locations.values || [],
          channels: channels.values || [],
          customers: customers.values || []
        });
      } catch (error) {
        console.error('Error loading dimension values:', error);
      }
    };

    loadDimensionValues();
  }, []);

  // Load drill options when level changes
  useEffect(() => {
    const loadDrillOptions = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/drill-options/${currentLevel}`);
        if (response.ok) {
          const options = await response.json();
          setDrillOptions(options);
        } else {
          // If drill options fail, just set empty options
          setDrillOptions({ drill_down: [], drill_up: [] });
        }
      } catch (error) {
        console.error('Error loading drill options:', error);
        setDrillOptions({ drill_down: [], drill_up: [] });
      }
    };

    loadDrillOptions();
  }, [currentLevel]);

  const handleChange = (param, value) => {
    setParams(prev => ({ ...prev, [param]: value }));
    if (param === 'level') {
      setCurrentLevel(value);
    }
  };

  const handleAggregate = () => {
    // Clean up params - only send fields that have values and are relevant
    const cleanParams = {
      level: params.level,
      forecast_days: params.forecast_days,
      include_promotional: true,
      include_base_demand: true
    };

    // Only add filter params if they have values
    if (params.brand) cleanParams.brand = params.brand;
    if (params.location) cleanParams.location = params.location;
    if (params.channel) cleanParams.channel = params.channel;
    if (params.customer) cleanParams.customer = params.customer;

    console.log('Sending aggregation request:', cleanParams);
    onAggregate(cleanParams);
  };

  const handleDrill = (direction, targetLevel) => {
    const currentFilters = {};
    
    // Only include filters that have actual values
    if (params.brand) currentFilters.brand = params.brand;
    if (params.location) currentFilters.location = params.location;
    if (params.channel) currentFilters.channel = params.channel;
    if (params.customer) currentFilters.customer = params.customer;

    const drillRequest = {
      from_level: currentLevel,
      to_level: targetLevel,
      direction: direction,
      current_filters: currentFilters,
      target_dimension: direction === 'down' ? 'brand' : 'location',
      forecast_days: params.forecast_days
    };

    console.log('Sending drill request:', drillRequest);
    onDrillDown(drillRequest);
    setCurrentLevel(targetLevel);
    setParams(prev => ({ ...prev, level: targetLevel }));
  };

  return (
    <div className="control-panel card">
      <div className="panel-header">
        <h3 className="panel-title">Multi-Level Aggregation</h3>
        <div className="level-indicator">
          Current Level: <span className="current-level">{params.level.replace('_', ' ').toUpperCase()}</span>
        </div>
      </div>

      <div className="tab-content">
        {/* Aggregation Level Selection */}
        <div className="form-section">
          <h4 className="section-title">Aggregation Level</h4>
          <div className="form-grid grid-2">
            <div className="form-group">
              <label className="form-label">Target Level</label>
              <select
                value={params.level}
                onChange={(e) => handleChange('level', e.target.value)}
                className="form-input"
              >
                {aggregationLevels.map(level => (
                  <option key={level.value} value={level.value}>
                    {level.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Forecast Days</label>
              <input
                type="number"
                value={params.forecast_days}
                onChange={(e) => handleChange('forecast_days', parseInt(e.target.value) || 30)}
                className="form-input"
                min="1"
                max="90"
              />
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="form-section">
          <h4 className="section-title">Filters</h4>
          <div className="form-grid grid-3">
            <div className="form-group">
              <label className="form-label">Brand {dimensionValues.brands.length > 0 && `(${dimensionValues.brands.length} available)`}</label>
              <select
                value={params.brand}
                onChange={(e) => handleChange('brand', e.target.value)}
                className="form-input"
              >
                <option value="">All Brands</option>
                {dimensionValues.brands.map(brand => (
                  <option key={brand} value={brand}>{brand}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Location {dimensionValues.locations.length > 0 && `(${dimensionValues.locations.length} available)`}</label>
              <select
                value={params.location}
                onChange={(e) => handleChange('location', e.target.value)}
                className="form-input"
              >
                <option value="">All Locations</option>
                {dimensionValues.locations.map(location => (
                  <option key={location} value={location}>{location}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Channel {dimensionValues.channels.length > 0 && `(${dimensionValues.channels.length} available)`}</label>
              <select
                value={params.channel}
                onChange={(e) => handleChange('channel', e.target.value)}
                className="form-input"
              >
                <option value="">All Channels</option>
                {dimensionValues.channels.length === 0 ? (
                  <option disabled>No channels found in dataset</option>
                ) : (
                  dimensionValues.channels.map(channel => (
                    <option key={channel} value={channel}>{channel}</option>
                  ))
                )}
              </select>
            </div>
          </div>
          
          {/* Customer filter in separate row if needed */}
          {dimensionValues.customers.length > 0 && (
            <div className="form-grid grid-1" style={{ marginTop: '10px' }}>
              <div className="form-group">
                <label className="form-label">Customer ({dimensionValues.customers.length} available)</label>
                <select
                  value={params.customer}
                  onChange={(e) => handleChange('customer', e.target.value)}
                  className="form-input"
                >
                  <option value="">All Customers</option>
                  {dimensionValues.customers.map(customer => (
                    <option key={customer} value={customer}>{customer}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="form-actions">
          <button
            onClick={handleAggregate}
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Aggregating...' : 'Generate Aggregated Forecast'}
          </button>

          {drillOptions.drill_up && drillOptions.drill_up.length > 0 && (
            <div className="drill-buttons">
              <span className="drill-label">Roll Up:</span>
              {drillOptions.drill_up.map(level => (
                <button
                  key={level}
                  onClick={() => handleDrill('up', level)}
                  className="btn btn-secondary btn-sm"
                  disabled={loading}
                >
                  ↑ {level.replace('_', ' ')}
                </button>
              ))}
            </div>
          )}

          {drillOptions.drill_down && drillOptions.drill_down.length > 0 && (
            <div className="drill-buttons">
              <span className="drill-label">Drill Down:</span>
              {drillOptions.drill_down.map(level => (
                <button
                  key={level}
                  onClick={() => handleDrill('down', level)}
                  className="btn btn-secondary btn-sm"
                  disabled={loading}
                >
                  ↓ {level.replace('_', ' ')}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Quick Navigation */}
        <div className="quick-nav">
          <h4 className="section-title">Quick Aggregate</h4>
          <div className="quick-buttons">
            <button
              onClick={() => handleChange('level', 'national_day')}
              className="btn btn-outline"
            >
              National
            </button>
            <button
              onClick={() => handleChange('level', 'brand_day')}
              className="btn btn-outline"
            >
              Brand
            </button>
            <button
              onClick={() => handleChange('level', 'brand_location_day')}
              className="btn btn-outline"
            >
              Brand-Location
            </button>
            <button
              onClick={() => handleChange('level', 'brand_channel_day')}
              className="btn btn-outline"
            >
              Brand-Channel
            </button>
          </div>
        </div>

        {/* Debug Info */}
        {process.env.NODE_ENV === 'development' && (
          <div className="debug-section" style={{ marginTop: '20px', padding: '10px', backgroundColor: '#f5f5f5', fontSize: '12px' }}>
            <h5>Debug Info:</h5>
            <p>API Base URL: {API_BASE_URL}</p>
            <p>Brands loaded: {dimensionValues.brands.length}</p>
            <p>Locations loaded: {dimensionValues.locations.length}</p>
            <p>Channels loaded: {dimensionValues.channels.length}</p>
            <p>Customers loaded: {dimensionValues.customers.length}</p>
            <p>Current level: {currentLevel}</p>
            {dimensionValues.channels.length === 0 && (
              <p style={{ color: 'red' }}>⚠️ No channels found - check backend column mapping</p>
            )}
            {dimensionValues.brands.length > 0 && (
              <p style={{ color: 'green' }}>Expected brands: Huggies, Kleenex, Scott, Kotex, Depend, Poise</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AggregationControl;