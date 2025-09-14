import React from 'react';
import PropTypes from 'prop-types';

const StatsDashboard = ({ forecastData }) => {
  if (!forecastData || !forecastData.values || forecastData.values.length === 0) {
    return (
      <div className="stats-dashboard">
        <div className="stat-card">
          <h3>No Data Available</h3>
          <p>Please provide forecast data</p>
        </div>
      </div>
    );
  }
  // Add error handling for missing data
  if (!forecastData || !forecastData.values || forecastData.values.length === 0) {
    return (
      <div className="stats-dashboard">
        <div className="stat-card">
          <h3>No Data Available</h3>
          <p>Please provide forecast data</p>
        </div>
      </div>
    );
  }

  const totalRevenue = forecastData.values.reduce((acc, curr) => acc + curr, 0);
  const avgRevenue = totalRevenue / forecastData.values.length;
  const peakRevenue = Math.max(...forecastData.values);
  const minRevenue = Math.min(...forecastData.values);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  return (
    <div className="stats-dashboard">
      <div className="stat-card">
        <h3>Total Revenue</h3>
        <p>{formatCurrency(totalRevenue)}</p>
      </div>
      <div className="stat-card">
        <h3>Average Revenue</h3>
        <p>{formatCurrency(avgRevenue)}</p>
      </div>
      <div className="stat-card">
        <h3>Peak Revenue</h3>
        <p>{formatCurrency(peakRevenue)}</p>
      </div>
      <div className="stat-card">
        <h3>Minimum Revenue</h3>
        <p>{formatCurrency(minRevenue)}</p>
      </div>
    </div>
  );
};

StatsDashboard.propTypes = {
  forecastData: PropTypes.shape({
    values: PropTypes.arrayOf(PropTypes.number)
  })
};

export default StatsDashboard;