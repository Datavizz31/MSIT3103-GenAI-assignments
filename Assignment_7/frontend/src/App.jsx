import React, { useState } from 'react';
import './App.css';

// Use the current host's port for the API
const API_URL = `${window.location.protocol}//${window.location.host}/generate/`;

function App() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    setMessages([...messages, { sender: 'user', text: prompt }]);
    setLoading(true);
    const startTime = performance.now();  // Record start time
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_length: 80, temperature: 0.5, num_return_sequences: 1 })
      });
      const data = await response.json();
      const endTime = performance.now();  // Record end time
      const clientLatency = (endTime - startTime) / 1000;  // Convert to seconds
      const displayLatency = ` ${clientLatency.toFixed(2)}s`;
      setMessages((msgs) => [
        ...msgs,
        { 
          sender: 'ai', 
          text: data.outputs && data.outputs[0] ? data.outputs[0] : 'No response.',
          latency: displayLatency
        }
      ]);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: 'ai', text: 'Error: Could not reach backend.' }
      ]);
    }
    setPrompt('');
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <h2>Generative AI Chat</h2>
      <div className="chat-box">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message-wrapper ${msg.sender}`}>
            <div className={`chat-message ${msg.sender}`}>{msg.text}</div>
            {msg.latency && <div className="chat-latency">{msg.latency}</div>}
          </div>
        ))}
        {loading && <div className="chat-message ai">Generating...</div>}
      </div>
      <form className="chat-form" onSubmit={handleSend}>
        <input
          type="text"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Type your prompt..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !prompt.trim()}>Send</button>
      </form>
    </div>
  );
}

export default App;
