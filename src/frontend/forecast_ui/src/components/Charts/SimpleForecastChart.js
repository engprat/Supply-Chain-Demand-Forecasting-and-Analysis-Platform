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

const SimpleForecastChart = ({ data }) => {
  if (!data) {
    return (
      <div className="chart-container">
        <h3 className="chart-title">📈 Forecast Chart</h3>
        <p className="no-data">No forecast data available</p>
      </div>
    );
  }

  const chartData = {
    labels: data.labels,
    datasets: [
      {
        label: 'Forecast',
        data: data.values,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
        tension: 0.4,
      },
      ...(data.confidence_upper ? [{
        label: 'Confidence Upper',
        data: data.confidence_upper,
        borderColor: 'rgba(255, 206, 86, 0.5)',
        backgroundColor: 'transparent',
        fill: false,
        borderDash: [5, 5],
      }] : []),
      ...(data.confidence_lower ? [{
        label: 'Confidence Lower',
        data: data.confidence_lower,
        borderColor: 'rgba(255, 206, 86, 0.5)',
        backgroundColor: 'transparent',
        fill: false,
        borderDash: [5, 5],
      }] : [])
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
        <h3 className="chart-title">📈 Forecast Chart</h3>
      </div>
      <div className="chart-wrapper">
        <Line data={chartData} options={chartOptions} />
      </div>
      {data.last_updated && (
        <div className="chart-footer">
          <small>Last updated: {new Date(data.last_updated).toLocaleString()}</small>
        </div>
      )}
    </div>
  );
};

export default SimpleForecastChart;
