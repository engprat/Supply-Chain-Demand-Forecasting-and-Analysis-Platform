import React, { useState, useEffect, useRef } from 'react';
import './ChatBotWindow.css';

const ChatBotWindow = ({ onClose, forecastData }) => {
  const [sessionId, setSessionId] = useState(null);

  // Format time to show only hour and minute
  const formatTime = (time) => {
    if (!time) {
      return new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: true
      });
    }
    
    const timeObj = typeof time === 'string' ? new Date(time) : time;
    if (isNaN(timeObj.getTime())) {
      return new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: true
      });
    }
    
    return timeObj.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true
    });
  };

  const [messages, setMessages] = useState([
    {
      content: "Hi there! I'm KayCee. Feel free to ask me any questions to get easier insights into your forecasts.",
      sender: 'bot',
      timestamp: formatTime(new Date())
    }
  ]);

  // Keep full message history but only show last 4 exchanges
  const getVisibleMessages = () => {
    const visibleMessages = [];
    let currentExchange = [];
    let exchanges = 0;

    // Start from the end and work backwards
    for (let i = messages.length - 1; i >= 0; i--) {
      currentExchange.unshift(messages[i]);
      
      if ((currentExchange.length === 2 || i === 0) && exchanges < 4) {
        visibleMessages.unshift(...currentExchange);
        exchanges++;
        currentExchange = [];
      }
    }

    return visibleMessages;
  };

  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    const createSession = async () => {
      try {
        // Get the current URL to determine the environment
        const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        const apiBaseUrl = isLocalhost 
          ? 'http://localhost:8000' 
          : window.location.origin;

        // Calculate total forecasted revenue if forecast data is available
        const forecastValues = forecastData?.forecast || [];
        const totalForecastedRevenue = forecastValues.reduce((sum, value) => sum + (Number(value) || 0), 0);
        
        // Calculate summary statistics
        const forecastStats = forecastValues.length > 0 ? {
          min: Math.min(...forecastValues),
          max: Math.max(...forecastValues),
          avg: forecastValues.reduce((a, b) => a + b, 0) / forecastValues.length
        } : null;

        // Prepare comprehensive metadata with all frontend-calculated data
        const metadata = {
          // Core forecast data
          forecast_data: forecastData,
          
          // Historical data if available
          historical_data: forecastData?.historical_data || [],
          
          // Model configuration and parameters
          model_config: {
            forecast_days: forecastData?.forecast_days,
            confidence_level: forecastData?.confidence_level,
            last_updated: forecastData?.last_updated,
            model_version: forecastData?.model_version
          },
          
          // Time series data
          time_series: {
            dates: forecastData?.dates || [],
            actuals: forecastData?.actuals || [],
            forecast: forecastValues,
            upper_bounds: forecastData?.upper_bounds || [],
            lower_bounds: forecastData?.lower_bounds || []
          },
          
          // Performance metrics if available
          metrics: {
            ...(forecastData?.metrics || {}),
            total_forecasted_revenue: totalForecastedRevenue,
            forecast_stats: forecastStats
          },
          
          // Additional context
          context: {
            current_time: new Date().toISOString(),
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            window_size: {
              width: window.innerWidth,
              height: window.innerHeight
            },
            // Add summary information for quick reference
            summary: {
              total_forecasted_revenue: totalForecastedRevenue,
              forecast_period: forecastData?.dates?.length ? {
                start: forecastData.dates[0],
                end: forecastData.dates[forecastData.dates.length - 1],
                days: forecastData.dates.length
              } : null
            }
          }
        };

        const response = await fetch(`${apiBaseUrl}/chatbot/sessions/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ metadata }),
        });

        if (!response.ok) {
          throw new Error('Failed to create chat session');
        }

        const data = await response.json();
        setSessionId(data.session_id);
      } catch (error) {
        console.error('Session creation error:', error);
        const errorMessage = {
          content: "Sorry, I'm having trouble connecting. Please try again later.",
          sender: 'bot',
          timestamp: formatTime(new Date())
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    };

    createSession();
  }, [forecastData]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !sessionId) return;

    const userMessage = {
      content: inputValue,
      sender: 'user',
      timestamp: formatTime(new Date())
    };

    setMessages(prev => [...prev, userMessage]);
    const messageToSend = inputValue;
    setInputValue('');
    setIsTyping(true);

    try {
      const response = await fetch(`http://localhost:8000/chatbot/sessions/${sessionId}/message` , {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: messageToSend }),
      });

      if (!response.ok) {
        throw new Error('Failed to get chat response');
      }

      const data = await response.json();
      
      const botResponse = {
        content: data.text,
        sender: 'bot',
        timestamp: formatTime(new Date())
      };
      
      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        content: "Sorry, I'm having trouble responding right now. Please try again.",
        sender: 'bot',
        timestamp: formatTime(new Date())
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-window-container">
      <div className="chat-window-header">
        <h3>KayCee</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      <div className="chat-messages">
        {getVisibleMessages().map((message, index) => (
          <div 
            key={index} 
            className={`message ${message.sender === 'user' ? 'user' : 'bot'}`}
          >
            <div className="message-content">
              <p>{message.content}</p>
              <span className="timestamp">{formatTime(new Date(message.timestamp))}</span>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="typing-indicator">
            <span>typing...</span>
            <div className="dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your message..."
          onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
        />
        <button onClick={handleSendMessage} disabled={!inputValue.trim()}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatBotWindow;
