import React from 'react';
import ConnectionTest from '../ConnectionTest/ConnectionTest';
import './Layout.css';

const Layout = ({
  activeWindow,
  setActiveWindow,
  connectionStatus,
  setConnectionStatus,
  setError,
  children
}) => {
  const navItems = [
    { id: 'dashboard', label: ' Dashboard' },
    { id: 'simulation', label: ' Simulation' },
    { id: 'promotional', label: ' Promotional Analysis' },
    { id: 'adhoc-forecast', label: ' Ad-Hoc Forecast' },
    { id: 'aggregation', label: 'Multi-Level Aggregation'},
    { id: 'customerOrders', label: ' Customer Orders (ML)' },

  ];

  return (
    <div className="app-layout">
      <nav className="side-nav">
        
        <ul className="nav-list">
          {navItems.map((item) => (
            <li
              key={item.id}
              className={`nav-item ${activeWindow === item.id ? 'active' : ''}`}
              onClick={() => setActiveWindow(item.id)}
            >
              {item.label}
            </li>
          ))}
        </ul>
        <div className="nav-footer">
          <ConnectionTest
            connectionStatus={connectionStatus}
            setConnectionStatus={setConnectionStatus}
            setError={setError}
            compact={true}
          />
        </div>
      </nav>
      <main className="main-content">
        {children}
      </main>
    </div>
  );
};

export default Layout;