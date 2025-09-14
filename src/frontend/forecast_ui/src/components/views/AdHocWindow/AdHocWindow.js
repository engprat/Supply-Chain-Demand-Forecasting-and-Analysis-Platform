import React, { useState } from 'react';
import AdHocForecastControl from '../../ControlPanel/AdHocForecastControl';

const AdHocWindow = ({ onAdHocAdjustment, loading, dataSummary, adHocData }) => {
  const [currentForecastLoading, setCurrentForecastLoading] = useState(false);
  const [suggestedForecast, setSuggestedForecast] = useState(null);

  // Function to fetch current forecast from backend
  const fetchCurrentForecast = async (sku_id, channel, location, target_date) => {
    if (!sku_id) return;
    
    setCurrentForecastLoading(true);
    try {
      const params = new URLSearchParams({
        sku_id,
        ...(channel && { channel }),
        ...(location && { location }),
        ...(target_date && { target_date })
      });
      
      const response = await fetch(`http://localhost:8000/api/current-forecast?${params}`);
      if (response.ok) {
        const data = await response.json();
        setSuggestedForecast(data);
        return data.current_forecast;
      }
    } catch (error) {
      console.error('Error fetching current forecast:', error);
    } finally {
      setCurrentForecastLoading(false);
    }
    return null;
  };

  return (
    <div className="window-container">
      <div className="window-header">
      </div>
      
      <div className="window-content">
        {/* Helper section for current forecast lookup */}
        {suggestedForecast && (
          <div className="forecast-helper">
            <div className="helper-header">
              <h3 className="helper-title">📊 Current Forecast Information</h3>
            </div>
            <div className="helper-content">
              <div className="forecast-info-grid">
                <div className="info-item">
                  <label>System Forecast:</label>
                  <span className="forecast-value">
                    {suggestedForecast.current_forecast || 'N/A'}
                  </span>
                </div>
                {suggestedForecast.sku_info?.sku_description && (
                  <div className="info-item">
                    <label>Product:</label>
                    <span>{suggestedForecast.sku_info.sku_description}</span>
                  </div>
                )}
                {suggestedForecast.sku_info?.category && (
                  <div className="info-item">
                    <label>Category:</label>
                    <span>{suggestedForecast.sku_info.category}</span>
                  </div>
                )}
                {suggestedForecast.validation?.warnings?.length > 0 && (
                  <div className="info-item warnings">
                    <label>Warnings:</label>
                    <ul>
                      {suggestedForecast.validation.warnings.map((warning, idx) => (
                        <li key={idx}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        
        <AdHocForecastControl
          onAdHocAdjustment={onAdHocAdjustment}
          loading={loading}
          dataSummary={dataSummary}
          onFetchCurrentForecast={fetchCurrentForecast}
          currentForecastLoading={currentForecastLoading}
          suggestedForecast={suggestedForecast?.current_forecast}
        />
        
        {adHocData && (
          <div className="results-section">
            <div className="results-header">
              <h3 className="results-title">✅ Adjustment Applied Successfully</h3>
            </div>
            <div className="results-content">
              <div className="adjustment-summary">
                <div className="summary-item">
                  <label>Adjustment ID:</label>
                  <span className="adjustment-id">{adHocData.adjustment_id}</span>
                </div>
                <div className="summary-item">
                  <label>SKU ID:</label>
                  <span>{adHocData.sku_id}</span>
                </div>
                <div className="summary-item">
                  <label>Date:</label>
                  <span>{adHocData.target_date}</span>
                </div>
                <div className="summary-item">
                  <label>Previous Value:</label>
                  <span>{adHocData.previous_value}</span>
                </div>
                <div className="summary-item">
                  <label>New Value:</label>
                  <span className="new-value">{adHocData.new_value}</span>
                </div>
                <div className="summary-item">
                  <label>Change:</label>
                  <span className={`change-value ${adHocData.change >= 0 ? 'positive' : 'negative'}`}>
                    {adHocData.change >= 0 ? '+' : ''}{adHocData.change}
                    {adHocData.change_percentage && 
                      ` (${adHocData.change_percentage >= 0 ? '+' : ''}${adHocData.change_percentage.toFixed(1)}%)`
                    }
                  </span>
                </div>
                <div className="summary-item">
                  <label>Adjustment Type:</label>
                  <span>{adHocData.adjustment_type}</span>
                </div>
                <div className="summary-item">
                  <label>Reason:</label>
                  <span>{adHocData.reason_category}</span>
                </div>
                {adHocData.notes && (
                  <div className="summary-item full-width">
                    <label>Notes:</label>
                    <span>{adHocData.notes}</span>
                  </div>
                )}
                
                {/* System comparison */}
                {adHocData.system_forecast_comparison && (
                  <>
                    <div className="summary-item">
                      <label>System Forecast:</label>
                      <span>{adHocData.system_forecast_comparison.system_forecast}</span>
                    </div>
                    <div className="summary-item">
                      <label>vs System:</label>
                      <span className={`change-value ${adHocData.system_forecast_comparison.adjustment_vs_system >= 0 ? 'positive' : 'negative'}`}>
                        {adHocData.system_forecast_comparison.adjustment_vs_system >= 0 ? '+' : ''}
                        {adHocData.system_forecast_comparison.adjustment_vs_system.toFixed(1)}
                      </span>
                    </div>
                  </>
                )}
                
                {/* Promotional context */}
                {adHocData.promotional_context?.active_promotion && (
                  <div className="summary-item full-width promotional-context">
                    <label>🎯 Active Promotion:</label>
                    <span>
                      {adHocData.promotional_context.promo_details.promo_name || 'Promotional period'}
                      {adHocData.promotional_context.promo_details.discount_percentage && 
                        ` (${adHocData.promotional_context.promo_details.discount_percentage}% off)`
                      }
                    </span>
                  </div>
                )}
                
                {/* Validation warnings */}
                {adHocData.validation_warnings?.length > 0 && (
                  <div className="summary-item full-width warnings">
                    <label>⚠️ Warnings:</label>
                    <ul>
                      {adHocData.validation_warnings.map((warning, idx) => (
                        <li key={idx}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <div className="summary-item">
                  <label>Applied At:</label>
                  <span>{new Date(adHocData.applied_at).toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdHocWindow;