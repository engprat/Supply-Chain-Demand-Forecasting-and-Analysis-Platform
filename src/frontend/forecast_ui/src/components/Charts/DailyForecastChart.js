import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import './Charts.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend
);

const DailyForecastChart = ({ forecastData, granularity }) => {
  if (!forecastData || !forecastData.forecasts || forecastData.forecasts.length === 0) {
    return (
      <div className="chart-container">
        <h3 className="chart-title">📊 Daily Forecast - {granularity}</h3>
        <p className="no-data">No forecast data available</p>
      </div>
    );
  }

  const forecast = forecastData.forecasts[0];
  
  const chartData = {
    labels: forecast.dates.map(date => {
      const d = new Date(date);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }),
    datasets: [
      {
        label: 'Base Forecast',
        data: forecast.base_forecast,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.1)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'Promotional Forecast',
        data: forecast.promotional_forecast,
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'Total Forecast',
        data: forecast.total_forecast,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
        tension: 0.4,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1000,
      easing: 'easeInOutQuart'
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          padding: 20,
          usePointStyle: true,
          pointStyle: 'circle',
          boxWidth: 8
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        titleColor: '#1a1a1a',
        bodyColor: '#4a4a4a',
        borderColor: '#e5e7eb',
        borderWidth: 1,
        padding: 12,
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
      },
    },
    scales: {
      x: {
        grid: {
          display: false
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      },
      y: {
        beginAtZero: true,
        grid: {
          drawBorder: false
        },
        title: {
          display: true,
          text: 'Demand Quantity',
          font: {
            weight: 'bold'
          }
        }
      }
    }
  };

  return (
    <div className="chart-container card">
      <div className="chart-header">
        <h3 className="chart-title">📊 Daily Forecast - {granularity}</h3>
      </div>
      <div className="chart-wrapper">
        <Line data={chartData} options={chartOptions} />
      </div>
      {forecast.has_historical_data && (
        <div className="historical-stats grid-3">
          <div className="stat-card blue">
            <p className="stat-label">Historical Base Avg</p>
            <p className="stat-value">{forecast.historical_base_avg?.toFixed(1) || 'N/A'}</p>
          </div>
          <div className="stat-card red">
            <p className="stat-label">Historical Promo Avg</p>
            <p className="stat-value">{forecast.historical_promo_avg?.toFixed(1) || 'N/A'}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DailyForecastChart;
