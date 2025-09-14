import { render, screen } from '@testing-library/react';
import App from './App';

test('renders forecast chart', () => {
  render(<App />);
  const linkElement = screen.getByText(/Revenue Forecast/i);
  expect(linkElement).toBeInTheDocument();
});

test('renders chatbot', () => {
  render(<App />);
  const linkElement = screen.getByText(/Ask me about forecasts/i);
  expect(linkElement).toBeInTheDocument();
});
