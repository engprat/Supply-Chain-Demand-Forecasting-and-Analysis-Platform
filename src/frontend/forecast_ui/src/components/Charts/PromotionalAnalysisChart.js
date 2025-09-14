import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import './Charts.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const PromotionalAnalysisChart = ({ promotionalData }) => {
  if (!promotionalData) {
    return (
      <div className="chart-container">
        <h3 className="chart-title">🎯 Promotional vs Base Demand</h3>
        <p className="no-data">No promotional data available</p>
      </div>
    );
  }

  const chartData = {
    labels: promotionalData.dates.map(date => {
      const d = new Date(date);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }),
    datasets: [
      {
        label: 'Base Demand',
        data: promotionalData.base_demand_forecast,
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
      {
        label: 'Promotional Demand',
        data: promotionalData.promotional_demand_forecast,
        backgroundColor: 'rgba(255, 99, 132, 0.8)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
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
        stacked: true,
        grid: {
          display: false
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      },
      y: {
        stacked: true,
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
        <h3 className="chart-title">🎯 Promotional vs Base Demand</h3>
      </div>
      <div className="chart-wrapper">
        <Bar data={chartData} options={chartOptions} />
      </div>
      <div className="promotional-summary grid-2">
        <div className="stat-card blue">
          <p className="stat-label">Total Base Demand</p>
          <p className="stat-value">{promotionalData.historical_summary?.total_base_demand?.toFixed(0) || 'N/A'}</p>
        </div>
        <div className="stat-card red">
          <p className="stat-label">Total Promotional Demand</p>
          <p className="stat-value">{promotionalData.historical_summary?.total_promotional_demand?.toFixed(0) || 'N/A'}</p>
        </div>
      </div>
    </div>
  );
};

export default PromotionalAnalysisChart;
