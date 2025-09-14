import React, { useEffect, useCallback, useRef } from 'react';
import './ConnectionTest.css';

const BACKEND_URL = 'http://localhost:8000';
const RETRY_DELAY = 5000; // 5 seconds between retries

const ConnectionTest = ({ connectionStatus, setConnectionStatus, setError, compact = false }) => {
  const retryTimeoutRef = useRef(null);
  const isMounted = useRef(true);

  const testConnection = useCallback(async (isRetry = false) => {
    if (!isRetry && connectionStatus === 'connected') {
      return; // Skip if we're already connected and this isn't a retry
    }

    try {
      setConnectionStatus('connecting');
      const response = await fetch(`${BACKEND_URL}/health`);
      
      if (!isMounted.current) return; // Skip if component unmounted
      
      if (response.ok) {
        const data = await response.json();
        setConnectionStatus('connected');
        setError(null);
        console.log('✅ Backend connection successful:', data);
        return data;
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (err) {
      if (!isMounted.current) return; // Skip if component unmounted
      
      console.error('❌ Backend connection failed:', err);
      setConnectionStatus('disconnected');
      setError(`Connection failed: ${err.message}`);
      
      // Schedule a retry
      if (isMounted.current) {
        clearTimeout(retryTimeoutRef.current);
        retryTimeoutRef.current = setTimeout(() => {
          testConnection(true);
        }, RETRY_DELAY);
      }
      
      return null;
    }
  }, [connectionStatus, setConnectionStatus, setError]);

  // Initial connection test on mount and when testConnection changes
  useEffect(() => {
    isMounted.current = true;
    testConnection();
    
    // Cleanup function
    return () => {
      isMounted.current = false;
      clearTimeout(retryTimeoutRef.current);
    };
  }, [testConnection]); // Include testConnection in dependencies

  if (compact) {
    return (
      <div className="connection-test compact">
        <div className={`status-indicator status-${connectionStatus}`}>
          {connectionStatus === 'connected' ? '✅ Connected' :
           connectionStatus === 'connecting' ? '🔄 Connecting...' :
           '❌ Disconnected'}
        </div>
      </div>
    );
  }

  return (
    <div className="connection-test card">
      <div className="connection-header">
        <div className="connection-info">
          <h3>🔌 Backend Connection</h3>
          <div className={`status-indicator status-${connectionStatus}`}>
            {connectionStatus === 'connected' ? '✅ Connected' :
             connectionStatus === 'connecting' ? '🔄 Testing...' :
             '❌ Disconnected'}
          </div>
        </div>
        <button
          onClick={() => testConnection()}
          className="btn btn-primary"
          disabled={connectionStatus === 'connecting'}
        >
          🔄 Test Connection
        </button>
      </div>
    </div>
  );
};

export default ConnectionTest;
