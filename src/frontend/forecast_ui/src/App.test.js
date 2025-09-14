import { render, screen } from '@testing-library/react';
import App from './App';

describe('App Component', () => {
  test('renders the main title', () => {
    render(<App />);
    const titleElement = screen.getByText(/Supply Chain Forecast Dashboard/i);
    expect(titleElement).toBeInTheDocument();
  });

  test('renders the connection test component', () => {
    render(<App />);
    const connectionHeader = screen.getByText(/🔌 Backend Connection/i);
    expect(connectionHeader).toBeInTheDocument();
  });

  test('renders the scenario planning section', () => {
    render(<App />);
    const scenarioTitle = screen.getByText(/🎯 Scenario Planning/i);
    expect(scenarioTitle).toBeInTheDocument();
  });
});
