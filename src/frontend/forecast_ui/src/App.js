import React, { useState, useEffect, useCallback } from 'react';
import { Chart as ChartJS } from 'chart.js';
import { registerables } from 'chart.js';

// Components
import Layout from './components/Layout/Layout';
import DashboardWindow from './components/views/DashboardWindow/DashboardWindow';
import SimulationWindow from './components/views/SimulationWindow/SimulationWindow';
import PromotionalWindow from './components/views/PromotionalWindow/PromotionalWindow';
import AdHocWindow from './components/views/AdHocWindow/AdHocWindow';
import AggregationWindow from './components/views/AggregationWindow/AggregationWindow';
import CustomerOrdersVizPanel from './components/views/CustomerOrdersWindow/CustomerOrdersVizPanel';

// Styles
import './App.css';
import './components/Charts/Charts.css';
import './components/ConnectionTest/ConnectionTest.css';
import './components/ControlPanel/DailyForecastControl.css';
import './components/ControlPanel/PromotionalAnalysisControl.css';
import './components/ControlPanel/AdHocForecastControl.css';
import './components/ScenarioPanel/ScenarioPanel.css';
import './components/Layout/Layout.css';
import './components/views/DashboardWindow/DashboardWindow.css';
import './components/views/SimulationWindow/SimulationWindow.css';
import './components/views/PromotionalWindow/PromotionalWindow.css';
import './components/views/AdHocWindow/AdHocWindow.css';
import './components/views/AggregationWindow/AggregationWindow.css';


// Register Chart.js components
ChartJS.register(...registerables, 'category', 'linear', 'point', 'line', 'bar', 'filler', 'title', 'tooltip', 'legend');

const BACKEND_URL = 'http://localhost:8000';

// Enhanced Top Header Component with Export Dropdown
const TopHeader = ({ onExport }) => {
  const [exportDropdownOpen, setExportDropdownOpen] = useState(false);

  const handleExportFormat = (format) => {
    onExport(format);
    setExportDropdownOpen(false);
  };

  return (
    <header className="app-header">
      <div className="header-container">
        <div className="header-content">
          <h1 className="app-title">Supply Chain Forecast Dashboard</h1>
          <div className="header-controls">
            <div className="export-dropdown">
              <button
                onClick={() => setExportDropdownOpen(!exportDropdownOpen)}
                className="btn btn-success dropdown-toggle"
              >
                📊 Export Data ▼
              </button>
              {exportDropdownOpen && (
                <div className="dropdown-menu">
                  <button
                    onClick={() => handleExportFormat('csv')}
                    className="dropdown-item"
                  >
                    📄 Export as CSV
                  </button>
                  <button
                    onClick={() => handleExportFormat('json')}
                    className="dropdown-item"
                  >
                    📋 Export as JSON
                  </button>
                  <button
                    onClick={() => handleExportFormat('xml')}
                    className="dropdown-item"
                  >
                    🔖 Export as XML
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

// Enhanced Export Functions Hook
const useExportFunctions = (activeWindow, dataSummary, dailyForecastData, simpleForecastData, promotionalData, adHocData, aggregationData, drillData) => {
  
  const generateExportData = useCallback(() => {
    const timestamp = new Date().toISOString().split('T')[0];
    let exportData = {};

    switch (activeWindow) {
      case 'dashboard':
        exportData = {
          window: 'dashboard',
          timestamp: timestamp,
          dataSummary: dataSummary ? {
            basicStats: {
              totalRecords: dataSummary.basic_stats?.total_records || 0,
              dateRange: {
                start: dataSummary.basic_stats?.date_range?.start || null,
                end: dataSummary.basic_stats?.date_range?.end || null
              }
            },
            granularityStats: {
              uniqueSkus: dataSummary.fr1_granularity_stats?.unique_skus || 0,
              uniqueChannels: dataSummary.fr1_granularity_stats?.unique_channels || 0,
              uniqueLocations: dataSummary.fr1_granularity_stats?.unique_locations || 0,
              uniqueCombinations: dataSummary.fr1_granularity_stats?.unique_sku_channel_location_combinations || 0
            },
            promotionalStats: {
              totalPromotionalRecords: dataSummary.fr2_promotional_stats?.total_promotional_records || 0,
              promotionalPercentage: dataSummary.fr2_promotional_stats?.promotional_percentage || 0
            }
          } : null,
          dailyForecast: dailyForecastData ? {
            granularityLevel: dailyForecastData.granularity_level,
            forecasts: dailyForecastData.forecasts?.map(forecast => ({
              skuId: forecast.sku_id,
              dates: forecast.dates,
              baseForecast: forecast.base_forecast,
              promotionalForecast: forecast.promotional_forecast,
              totalForecast: forecast.total_forecast,
              historicalBaseAvg: forecast.historical_base_avg,
              historicalPromoAvg: forecast.historical_promo_avg,
              trendSlope: forecast.trend_slope,
              hasHistoricalData: forecast.has_historical_data,
              forecastingMethod: forecast.forecasting_method,
              dataSource: forecast.data_source
            })) || []
          } : null
        };
        break;

      case 'simulation':
        exportData = {
          window: 'simulation',
          timestamp: timestamp,
          scenarioForecast: simpleForecastData ? {
            labels: simpleForecastData.labels,
            values: simpleForecastData.values,
            confidenceUpper: simpleForecastData.confidence_upper,
            confidenceLower: simpleForecastData.confidence_lower,
            featureCount: simpleForecastData.feature_count,
            featuresUsed: simpleForecastData.features_used,
            metadata: simpleForecastData.metadata
          } : null
        };
        break;

      case 'promotional':
        exportData = {
          window: 'promotional',
          timestamp: timestamp,
          promotionalAnalysis: promotionalData ? {
            dates: promotionalData.dates,
            baseDemandForecast: promotionalData.base_demand_forecast,
            promotionalDemandForecast: promotionalData.promotional_demand_forecast,
            historicalSummary: promotionalData.historical_summary,
            promotionalImpactAnalysis: promotionalData.promotional_impact_analysis,
            dataQuality: promotionalData.data_quality
          } : null
        };
        break;

      case 'adhoc-forecast':
        exportData = {
          window: 'adhoc-forecast',
          timestamp: timestamp,
          adHocAdjustment: adHocData ? {
            skuId: adHocData.sku_id,
            channel: adHocData.channel,
            location: adHocData.location,
            targetDate: adHocData.target_date,
            currentForecast: adHocData.current_forecast,
            newForecastValue: adHocData.new_forecast_value,
            adjustmentType: adHocData.adjustment_type,
            adjustmentValue: adHocData.adjustment_value,
            reasonCategory: adHocData.reason_category,
            notes: adHocData.notes,
            timestamp: adHocData.timestamp,
            adjustmentSummary: adHocData.adjustment_summary,
            dataContext: adHocData.data_context
          } : null
        };
        break;

      case 'aggregation':
        exportData = {
          window: 'aggregation',
          timestamp: timestamp,
          aggregationData: aggregationData ? {
            level: aggregationData.level,
            aggregationKey: aggregationData.aggregation_key,
            forecastData: aggregationData.forecast_data,
            summaryMetrics: aggregationData.summary_metrics,
            dataQuality: aggregationData.data_quality,
            drillDownOptions: aggregationData.drill_down_options
          } : null,
          drillData: drillData ? {
            targetLevel: drillData.target_level,
            drillDimension: drillData.drill_dimension,
            results: drillData.drill_results?.map(result => ({
              level: result.level,
              aggregationKey: result.aggregation_key,
              summaryMetrics: result.summary_metrics,
              dataQuality: result.data_quality,
              drillContext: result.drill_context
            })) || [],
            metadata: drillData.metadata
          } : null
        };
        break;

      default:
        exportData = {
          window: 'unknown',
          timestamp: timestamp,
          message: 'No data available for current window'
        };
    }

    return exportData;
  }, [activeWindow, dataSummary, dailyForecastData, simpleForecastData, promotionalData, adHocData, aggregationData, drillData]);

  const exportAsCSV = useCallback((data) => {
    const timestamp = new Date().toISOString().split('T')[0];
    
    const flattenObject = (obj, prefix = '') => {
      const flattened = [];
      for (const [key, value] of Object.entries(obj)) {
        const newKey = prefix ? `${prefix}.${key}` : key;
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          flattened.push(...flattenObject(value, newKey));
        } else if (Array.isArray(value)) {
          value.forEach((item, index) => {
            if (typeof item === 'object') {
              flattened.push(...flattenObject(item, `${newKey}[${index}]`));
            } else {
              flattened.push([`${newKey}[${index}]`, item]);
            }
          });
        } else {
          flattened.push([newKey, value]);
        }
      }
      return flattened;
    };

    const csvContent = [
      ['Field', 'Value'],
      ...flattenObject(data)
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${data.window}_export_${timestamp}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  }, []);

  const exportAsJSON = useCallback((data) => {
    const timestamp = new Date().toISOString().split('T')[0];
    const jsonContent = JSON.stringify(data, null, 2);
    
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${data.window}_export_${timestamp}.json`;
    link.click();
    window.URL.revokeObjectURL(url);
  }, []);

  const exportAsXML = useCallback((data) => {
    const timestamp = new Date().toISOString().split('T')[0];
    
    const objectToXML = (obj, indent = 0) => {
      const spaces = '  '.repeat(indent);
      let xml = '';
      
      for (const [key, value] of Object.entries(obj)) {
        const sanitizedKey = key.replace(/[^a-zA-Z0-9_]/g, '_');
        
        if (value === null || value === undefined) {
          xml += `${spaces}<${sanitizedKey} />\n`;
        } else if (Array.isArray(value)) {
          xml += `${spaces}<${sanitizedKey}>\n`;
          value.forEach((item, index) => {
            if (typeof item === 'object') {
              xml += `${spaces}  <item_${index}>\n`;
              xml += objectToXML(item, indent + 2);
              xml += `${spaces}  </item_${index}>\n`;
            } else {
              xml += `${spaces}  <item_${index}>${String(item)}</item_${index}>\n`;
            }
          });
          xml += `${spaces}</${sanitizedKey}>\n`;
        } else if (typeof value === 'object') {
          xml += `${spaces}<${sanitizedKey}>\n`;
          xml += objectToXML(value, indent + 1);
          xml += `${spaces}</${sanitizedKey}>\n`;
        } else {
          xml += `${spaces}<${sanitizedKey}>${String(value)}</${sanitizedKey}>\n`;
        }
      }
      return xml;
    };

    const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<export>
${objectToXML(data, 1)}
</export>`;

    const blob = new Blob([xmlContent], { type: 'application/xml' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${data.window}_export_${timestamp}.xml`;
    link.click();
    window.URL.revokeObjectURL(url);
  }, []);

  const handleExport = useCallback((format) => {
    const data = generateExportData();
    
    if (!data || (data.window === 'unknown')) {
      alert('No data available to export from current window');
      return;
    }

    switch (format) {
      case 'csv':
        exportAsCSV(data);
        break;
      case 'json':
        exportAsJSON(data);
        break;
      case 'xml':
        exportAsXML(data);
        break;
      default:
        alert('Invalid export format');
    }

    console.log(`✅ Exported ${data.window} data as ${format.toUpperCase()}`);
  }, [generateExportData, exportAsCSV, exportAsJSON, exportAsXML]);

  return handleExport;
};

// Error Display Component
const ErrorDisplay = ({ error, onClose }) => (
  error && (
    <div className="alert alert-error">
      <span className="alert-icon">⚠️</span>
      <p className="alert-message">{error}</p>
      <button className="alert-close" onClick={onClose}>×</button>
    </div>
  )
);

// CSS for dropdown styling
const dropdownStyles = `
.export-dropdown {
  position: relative;
  display: inline-block;
}

.dropdown-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  background: #28a745;
  color: white;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;
}

.dropdown-toggle:hover {
  background: #218838;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  right: 0;
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  min-width: 180px;
  margin-top: 4px;
}

.dropdown-item {
  display: block;
  width: 100%;
  padding: 12px 16px;
  border: none;
  background: none;
  text-align: left;
  cursor: pointer;
  font-size: 14px;
  color: #333;
  transition: background-color 0.2s;
}

.dropdown-item:hover {
  background: #f8f9fa;
}

.dropdown-item:first-child {
  border-top-left-radius: 4px;
  border-top-right-radius: 4px;
}

.dropdown-item:last-child {
  border-bottom-left-radius: 4px;
  border-bottom-right-radius: 4px;
}
`;

// Main App Component
const App = () => {
  // State
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [dataSummary, setDataSummary] = useState(null);
  const [dailyForecastData, setDailyForecastData] = useState(null);
  const [promotionalData, setPromotionalData] = useState(null);
  const [adHocData, setAdHocData] = useState(null);
  const [simpleForecastData, setSimpleForecastData] = useState(null);
  const [historicalData, setHistoricalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeWindow, setActiveWindow] = useState('dashboard');
  
  // Simplified model params - now only used for initial data loading
  const [modelParams] = useState({
    days: 30,
    startDate: new Date().toISOString().split('T')[0]
  });
  
  const [aggregationData, setAggregationData] = useState(null);
  const [drillData, setDrillData] = useState(null);

  // Enhanced export hook
  const handleExport = useExportFunctions(
    activeWindow, 
    dataSummary, 
    dailyForecastData, 
    simpleForecastData, 
    promotionalData, 
    adHocData, 
    aggregationData, 
    drillData
  );

  // Load initial data
  const loadDataSummary = useCallback(async () => {
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
  }, []);

  const loadSimpleForecast = useCallback(async () => {
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
  }, [modelParams.days]);

  // Initialize data on component mount
  useEffect(() => {
    if (connectionStatus === 'connected') {
      loadDataSummary();
      loadSimpleForecast();
    }
  }, [connectionStatus, loadDataSummary, loadSimpleForecast]);

  const handleDailyForecast = useCallback(async (params) => {
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
  }, []);

  const handlePromotionalForecast = useCallback(async (params) => {
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
  }, []);

  const handleAdHocAdjustment = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Applying ad-hoc forecast adjustment with params:', params);
      
      // Calculate the new forecast value based on adjustment type
      let newForecastValue;
      const current = parseFloat(params.current_forecast);
      const adjustment = parseFloat(params.adjustment_value);
      
      switch (params.adjustment_type) {
        case 'absolute':
          newForecastValue = adjustment;
          break;
        case 'percentage':
          newForecastValue = current * (1 + adjustment / 100);
          break;
        case 'multiplier':
          newForecastValue = current * adjustment;
          break;
        default:
          newForecastValue = current;
      }
      
      const response = await fetch(`${BACKEND_URL}/api/adhoc-adjustment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...params,
          new_forecast_value: Math.round(newForecastValue * 100) / 100,
          timestamp: new Date().toISOString()
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setAdHocData({
        ...params,
        new_forecast_value: Math.round(newForecastValue * 100) / 100,
        timestamp: new Date().toISOString(),
        ...data
      });
      console.log('✅ Ad-hoc adjustment applied:', data);
    } catch (err) {
      console.error('❌ Error applying ad-hoc adjustment:', err);
      setError(`Ad-hoc adjustment failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleScenarioAnalysis = useCallback(async ({ scenario, forecast_days = 30 }) => {
    setScenarioLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Running scenario analysis with params:', scenario);
      const response = await fetch(`${BACKEND_URL}/api/scenario-forecast`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          forecast_days: forecast_days, // Use the forecast_days from scenario params
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
  }, []);

  const handleAggregate = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    setDrillData(null); // Clear drill data when new aggregation is requested
    
    try {
      console.log('🔄 Generating aggregated forecast with params:', params);
      const response = await fetch(`${BACKEND_URL}/api/aggregate-forecast`, {
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
      setAggregationData(data);
      console.log('✅ Aggregated forecast generated:', data);
    } catch (err) {
      console.error('❌ Error generating aggregated forecast:', err);
      setError(`Aggregated forecast failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrillDown = useCallback(async (drillRequest) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Drilling down with params:', drillRequest);
      const response = await fetch(`${BACKEND_URL}/api/drill-down`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(drillRequest),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setDrillData(data);
      setAggregationData(null); // Clear single aggregation when drilling
      console.log('✅ Drill-down completed:', data);
    } catch (err) {
      console.error('❌ Error during drill-down:', err);
      setError(`Drill-down failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const renderContent = () => {
    switch (activeWindow) {
      case 'dashboard':
        return (
          <DashboardWindow
            onDailyForecast={handleDailyForecast}
            loading={loading}
            dataSummary={dataSummary}
            dailyForecastData={dailyForecastData}
          />
        );
      case 'simulation':
        return (
          <SimulationWindow
            onRunScenario={handleScenarioAnalysis}
            loading={scenarioLoading}
            simpleForecastData={simpleForecastData}
          />
        );
      case 'promotional':
        return (
          <PromotionalWindow
            onPromotionalForecast={handlePromotionalForecast}
            loading={loading}
            dataSummary={dataSummary}
            promotionalData={promotionalData}
          />
        );
      case 'adhoc-forecast':
        return (
          <AdHocWindow
            onAdHocAdjustment={handleAdHocAdjustment}
            loading={loading}
            dataSummary={dataSummary}
            adHocData={adHocData}
          />
        );
      case 'aggregation':
        return (
          <AggregationWindow
            onAggregate={handleAggregate}
            onDrillDown={handleDrillDown}
            loading={loading}
            aggregationData={aggregationData}
            drillData={drillData}
          />
        );

      case 'customerOrders':
        return <CustomerOrdersVizPanel />;


      default:
        return null;
    }
  };

  return (
    <div className="app">
      <style>{dropdownStyles}</style>
      <TopHeader onExport={handleExport} />
      <Layout
        activeWindow={activeWindow}
        setActiveWindow={setActiveWindow}
        connectionStatus={connectionStatus}
        setConnectionStatus={setConnectionStatus}
        setError={setError}
      >
        <ErrorDisplay 
          error={error} 
          onClose={() => setError(null)} 
        />
        {renderContent()}
      </Layout>
    </div>
  );
};

export default App;