import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import './ChatBotLauncher.css';

const ChatBotLauncher = ({ onToggleChat }) => {
  const [hasNewMessages, setHasNewMessages] = useState(false);

  return (
    <div className="chat-launcher-container">
      <button 
        className={`chat-launcher ${hasNewMessages ? 'has-messages' : ''}`} 
        aria-label="Open chat window"
        onClick={() => {
          onToggleChat();
          setHasNewMessages(false);
        }}
      >
        <span className="chat-icon">KC</span>
        {hasNewMessages && <span className="message-dot" />}
      </button>
    </div>
  );
};

ChatBotLauncher.propTypes = {
  onToggleChat: PropTypes.func.isRequired
};

export default ChatBotLauncher;
