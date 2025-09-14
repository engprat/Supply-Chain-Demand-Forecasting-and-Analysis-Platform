import React from 'react';
import PromotionalAnalysisControl from '../../ControlPanel/PromotionalAnalysisControl';
import PromotionalAnalysisChart from '../../Charts/PromotionalAnalysisChart';
import './PromotionalWindow.css';

const PromotionalWindow = ({
  onPromotionalForecast,
  loading,
  dataSummary,
  promotionalData
}) => {
  // Determine regional display preferences from promotional data
  const getRegionalDisplayMode = () => {
    if (!promotionalData?.regional_config) return 'standard';
    
    const { region, promo_modeling_approach, forecast_granularity } = promotionalData.regional_config;
    
    if (region === 'China' && promo_modeling_approach === 'embedded') {
      return 'china-embedded';
    } else if (region === 'Korea' && promo_modeling_approach === 'separated') {
      return 'korea-separated';
    } else {
      return 'standard';
    }
  };

  const renderRegionalAnalysisHeader = () => {
    if (!promotionalData?.regional_config) return null;
    
    const { region, forecast_granularity, promo_modeling_approach } = promotionalData.regional_config;
    
    return (
      <div className="regional-header">
        <div className="region-badge">
          <span className="region-name">{region}</span>
          <span className="region-config">
            {forecast_granularity} • {promo_modeling_approach}
          </span>
        </div>
      </div>
    );
  };

  const renderChinaEmbeddedView = () => {
    return (
      <>
        <section className="card china-view">
          <div className="section-header">
            <h2 className="section-title">🇨🇳 China Monthly Promotional Forecast</h2>
            {renderRegionalAnalysisHeader()}
          </div>
          <PromotionalAnalysisChart 
            promotionalData={promotionalData}
            displayMode="embedded"
            granularity="monthly"
          />
          
          <div className="china-insights">
            <div className="insight-grid">
              <div className="insight-card primary">
                <h4>📊 Monthly Combined Forecast</h4>
                <p>Promotional effects are embedded within the base forecast, providing a unified monthly view optimized for Chinese market planning cycles.</p>
              </div>
              
              <div className="insight-card secondary">
                <h4>🎯 Integrated Promotion Impact</h4>
                <p>Promotion timing and effects are automatically factored into demand patterns, reflecting typical Chinese consumer behavior and promotional calendars.</p>
              </div>
            </div>
          </div>
        </section>

        {promotionalData.monthly_summary && (
          <section className="monthly-summary card">
            <h3 className="section-title">📈 Monthly Performance Summary</h3>
            <div className="monthly-stats grid-4">
              <div className="stat-card blue">
                <p className="stat-label">📅 Forecast Months</p>
                <p className="stat-value">
                  {Math.ceil(promotionalData.forecast_days / 30)}
                </p>
              </div>
              <div className="stat-card green">
                <p className="stat-label">💰 Total Demand</p>
                <p className="stat-value">
                  {promotionalData.monthly_summary.total_monthly_demand?.toLocaleString() || 'N/A'}
                </p>
              </div>
              <div className="stat-card purple">
                <p className="stat-label">📈 Monthly Growth Rate</p>
                <p className="stat-value">
                  {promotionalData.monthly_summary.monthly_growth_rate?.toFixed(1) || '0'}%
                </p>
              </div>
              <div className="stat-card orange">
                <p className="stat-label">🎯 Promotion Intensity</p>
                <p className="stat-value">
                  {promotionalData.monthly_summary.promotion_intensity || 'Medium'}
                </p>
              </div>
            </div>
          </section>
        )}
      </>
    );
  };

  const renderKoreaSeparatedView = () => {
    return (
      <>
        <section className="card korea-view">
          <div className="section-header">
            <h2 className="section-title">🇰🇷 Korea Daily Promotional Analysis</h2>
            {renderRegionalAnalysisHeader()}
          </div>
          <PromotionalAnalysisChart 
            promotionalData={promotionalData}
            displayMode="separated"
            granularity="daily"
          />
          
          <div className="korea-insights">
            <div className="separated-forecast-grid">
              <div className="base-forecast-section">
                <h4>📊 Base Demand Forecast</h4>
                <div className="base-stats">
                  <div className="stat-item">
                    <span className="stat-label">Average Daily Base:</span>
                    <span className="stat-value">
                      {promotionalData.historical_summary?.historical_base_avg?.toLocaleString() || 'N/A'}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Base Trend:</span>
                    <span className="stat-value trend-positive">
                      ↗ Stable Growth
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="promo-forecast-section">
                <h4>🎯 Promotional Demand Forecast</h4>
                <div className="promo-stats">
                  <div className="stat-item">
                    <span className="stat-label">Average Promo Impact:</span>
                    <span className="stat-value">
                      {promotionalData.promotional_impact_analysis?.avg_promotional_lift?.toFixed(2) || '0'}x
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Promo Frequency:</span>
                    <span className="stat-value">
                      {((promotionalData.promotional_impact_analysis?.promotional_frequency || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="korea-detailed-analysis card">
          <h3 className="section-title">🔍 Detailed Promotional vs Base Analysis</h3>
          <div className="analysis-tabs">
            <div className="tab-content korea-comparison">
              <div className="comparison-grid">
                <div className="comparison-item">
                  <h5>Base Demand Characteristics</h5>
                  <ul>
                    <li>Consistent daily patterns</li>
                    <li>Predictable weekly cycles</li>
                    <li>Seasonal stability</li>
                  </ul>
                </div>
                <div className="comparison-item">
                  <h5>Promotional Impact Patterns</h5>
                  <ul>
                    <li>Event-driven spikes</li>
                    <li>Clear lift measurement</li>
                    <li>Post-promotion normalization</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>
      </>
    );
  };

  const renderStandardView = () => {
    return (
      <>
        <section className="card">
          <div className="section-header">
            <h2 className="section-title">🎯 Promotional Analysis</h2>
            {renderRegionalAnalysisHeader()}
          </div>
          <PromotionalAnalysisChart 
            promotionalData={promotionalData}
            displayMode="standard"
          />
        </section>

        {promotionalData.promotional_impact_analysis && (
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
      </>
    );
  };

  const renderDataQuality = () => {
    if (!promotionalData?.data_quality) return null;
    
    return (
      <section className="data-quality card">
        <h3 className="section-title">📊 Data Quality & Coverage</h3>
        <div className="quality-grid grid-3">
          <div className="quality-item">
            <h5>Historical Records</h5>
            <p className="quality-value">
              {promotionalData.data_quality.historical_records_used?.toLocaleString() || 'N/A'}
            </p>
          </div>
          <div className="quality-item">
            <h5>Promotional Records</h5>
            <p className="quality-value">
              {promotionalData.data_quality.promotional_records_found?.toLocaleString() || 'N/A'}
            </p>
          </div>
          <div className="quality-item">
            <h5>Date Range</h5>
            <p className="quality-value">
              {promotionalData.data_quality.date_range_analyzed ? 
                `${promotionalData.data_quality.date_range_analyzed.start} to ${promotionalData.data_quality.date_range_analyzed.end}` :
                'N/A'
              }
            </p>
          </div>
        </div>
      </section>
    );
  };

  const displayMode = getRegionalDisplayMode();

  return (
    <div className="promotional-window">
      <PromotionalAnalysisControl
        onPromotionalForecast={onPromotionalForecast}
        loading={loading}
        dataSummary={dataSummary}
      />
      
      {promotionalData && (
        <div className="promotional-results">
          {displayMode === 'china-embedded' && renderChinaEmbeddedView()}
          {displayMode === 'korea-separated' && renderKoreaSeparatedView()}
          {displayMode === 'standard' && renderStandardView()}
          
          {renderDataQuality()}
        </div>
      )}
      
      {!promotionalData && !loading && (
        <div className="empty-state card">
          <div className="empty-content">
            <h3>🎯 Ready for Promotional Analysis</h3>
            <p>Select your region and enter the required parameters above to begin analysis.</p>
            <div className="region-features">
              <div className="feature-item">
                <strong>🇨🇳 China:</strong> Monthly forecasts with embedded promotional effects
              </div>
              <div className="feature-item">
                <strong>🇰🇷 Korea:</strong> Daily forecasts with separated promo/base analysis
              </div>
              <div className="feature-item">
                <strong>🌍 Other Regions:</strong> Flexible granularity and modeling approaches
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PromotionalWindow;