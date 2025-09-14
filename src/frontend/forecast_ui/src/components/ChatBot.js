import React from 'react';
// No props to validate for ChatBot, but add future-proofing if needed.
import { Widget } from 'react-chat-widget';
import 'react-chat-widget/lib/styles.css';

const ChatBot = () => {
  const handleNewUserMessage = (newMessage) => {
    console.log('New message: ', newMessage);
    
    
  };

  return (
    <div>
      <Widget
        handleNewUserMessage={handleNewUserMessage}
        title="AI Supply Chain Assistant"
        subtitle="Ask me about forecasts, inventory, or supply chain insights"
      />
    </div>
  );
};

export default ChatBot;
