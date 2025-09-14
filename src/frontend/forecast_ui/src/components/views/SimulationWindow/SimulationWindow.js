import React from 'react';
import ScenarioPanel from '../../ScenarioPanel/ScenarioPanel';
import SimpleForecastChart from '../../Charts/SimpleForecastChart';
import './SimulationWindow.css';

const SimulationWindow = ({
  onRunScenario,
  loading,
  simpleForecastData
}) => {
  return (
    <div className="simulation-window">
      <ScenarioPanel onRunScenario={onRunScenario} loading={loading} />
      {simpleForecastData && (
        <section className="card">
          <h2 className="section-title">📈 Scenario Forecast Chart</h2>
          <SimpleForecastChart data={simpleForecastData} />
        </section>
      )}
    </div>
  );
};

export default SimulationWindow;
