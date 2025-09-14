import React from 'react';
import { render, screen } from '@testing-library/react';
import App from '../App';

// Simple smoke test to verify the app renders
it('renders without crashing', () => {
  render(<App />);
  expect(screen.getByText(/supply chain forecast dashboard/i)).toBeInTheDocument();
});
