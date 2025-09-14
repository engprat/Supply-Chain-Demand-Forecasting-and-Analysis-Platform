import React, { useEffect } from 'react';
import DailyForecastControl from '../../ControlPanel/DailyForecastControl';
import DailyForecastChart from '../../Charts/DailyForecastChart';
import './DashboardWindow.css';

const DashboardWindow = ({
  onDailyForecast,
  loading,
  dataSummary,
  dailyForecastData
}) => {
  useEffect(() => {
    // Trigger the daily forecast on initial render
    onDailyForecast({});
  }, [onDailyForecast]);

  return (
    <div className="dashboard-window">
      <DailyForecastControl
        onDailyForecast={onDailyForecast}
        loading={loading}
        dataSummary={dataSummary}
      />
      {dailyForecastData && (
        <section className="card">
          <h2 className="section-title">📊 Daily Forecast Analysis</h2>
          <DailyForecastChart
            forecastData={dailyForecastData}
            granularity={dailyForecastData.granularity_level}
          />
        </section>
      )}
    </div>
  );
};

export default DashboardWindow;
