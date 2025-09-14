import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Filler,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Filler,
  Title,
  Tooltip,
  Legend
);

const ForecastChart = ({ data }) => {
  const [highDemandData, setHighDemandData] = useState(null);
  const [monthlyPeakData, setMonthlyPeakData] = useState(null);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const highDemandRes = await fetch('http://localhost:8000/api/high-demand-products');
        const highDemandJson = await highDemandRes.json();
        setHighDemandData(highDemandJson);

        const monthlyPeakRes = await fetch('http://localhost:8000/api/monthly-peak-demand');
        const monthlyPeakJson = await monthlyPeakRes.json();
        setMonthlyPeakData(monthlyPeakJson);
      } catch (error) {
        console.error('Error fetching additional chart data:', error);
      }
    };

    fetchChartData();
  }, []);

  if (!data || !data.labels || !data.values || data.labels.length === 0) {
    return (
      <div className="chart-container">
        <h2>Revenue Forecast</h2>
        <p>No data available for chart</p>
      </div>
    );
  }

  const maxIndex = data.values.indexOf(Math.max(...data.values));
  const pointBackgroundColors = data.values.map((_, i) =>
    i === maxIndex ? 'rgba(255, 99, 132, 1)' : 'rgba(75,192,192,1)'
  );

  const forecastChartData = {
    labels: data.labels,
    datasets: [
      {
        label: 'Forecasted Revenue',
        data: data.values,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.1)',
        fill: 'origin',
        tension: 0.4,
        pointBackgroundColor: pointBackgroundColors,
        pointRadius: 4,
      },
      ...(data.confidence_upper && data.confidence_lower
        ? [
            {
              label: 'Confidence Interval',
              data: data.values,
              borderColor: 'rgba(0, 0, 0, 0)',
              backgroundColor: 'rgba(54, 162, 235, 0.2)',
              fill: {
                target: {
                  value: data.confidence_lower,
                },
                above: 'rgba(54, 162, 235, 0.2)',
              },
              pointRadius: 0,
              tension: 0.4,
            },
          ]
        : []),
    ],
  };

  const highDemandChartData = highDemandData && {
    labels: highDemandData.products,
    datasets: [
      {
        label: 'High Demand Products',
        data: highDemandData.demand,
        backgroundColor: 'rgba(255, 99, 132, 0.6)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
      },
    ],
  };

  const monthlyPeakChartData = monthlyPeakData && {
    labels: monthlyPeakData.months,
    datasets: [
      {
        label: 'Monthly Peak Demand',
        data: monthlyPeakData.peak_demand,
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1200,
      easing: 'easeOutQuart',
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        font: {
          size: 18,
        },
      },
    },
  };

  return (
    <div>
      <div className="chart-container" style={{ height: '400px' }}>
        <h2>Revenue Forecast</h2>
        <Line data={forecastChartData} options={{...options, plugins: {...options.plugins, title: {...options.plugins.title, text: 'Revenue Forecast Chart'}}, scales: {y: {beginAtZero: false, ticks: {callback: (value) => `${value.toLocaleString()}`}, title: {display: true, text: 'Revenue ($)'}}, x: {title: {display: true, text: 'Date'}}}}} />
      </div>
      {highDemandChartData && (
        <div className="chart-container" style={{ height: '400px', marginTop: '20px' }}>
          <h2>High Demand Products</h2>
          <Bar data={highDemandChartData} options={{...options, plugins: {...options.plugins, title: {...options.plugins.title, text: 'Top 10 High Demand Products'}}, scales: {y: {beginAtZero: true, title: {display: true, text: 'Total Demand'}}, x: {title: {display: true, text: 'Product'}}}}} />
        </div>
      )}
      {monthlyPeakChartData && (
        <div className="chart-container" style={{ height: '400px', marginTop: '20px' }}>
          <h2>Monthly Peak Demand</h2>
          <Line data={monthlyPeakChartData} options={{...options, plugins: {...options.plugins, title: {...options.plugins.title, text: 'Monthly Peak Product Demand'}}, scales: {y: {beginAtZero: true, title: {display: true, text: 'Peak Order Quantity'}}, x: {title: {display: true, text: 'Month'}}}}} />
        </div>
      )}
    </div>
  );
};

ForecastChart.propTypes = {
  data: PropTypes.shape({
    labels: PropTypes.array,
    values: PropTypes.array,
    confidence_upper: PropTypes.array,
    confidence_lower: PropTypes.array
  })
};

export default ForecastChart;



