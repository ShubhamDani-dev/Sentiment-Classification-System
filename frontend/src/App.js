import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, { 
        text: text.trim() 
      });
      setResult(response.data);
    } catch (err) {
      setResult({ error: 'Failed to analyze text' });
    }
    setLoading(false);
  };

  const fetchModelStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/model-status`);
      setModelStatus(response.data);
    } catch (err) {
      console.log('Could not fetch model status');
    }
  };

  React.useEffect(() => {
    fetchModelStatus();
  }, []);

  const getSentimentIcon = (label) => {
    return label === 'positive' ? '😊' : '😞';
  };

  const getConfidenceColor = (score) => {
    if (score > 0.8) return '#28a745';
    if (score > 0.6) return '#ffc107';
    return '#dc3545';
  };

  return (
    <div className="App">
      <div className="container">
        <div className="header">
          <h1>✨ Sentiment Analysis</h1>
          <p className="subtitle">Discover the emotional tone of your text using AI</p>
          {modelStatus && (
            <div className="model-status">
              <span className="status-indicator"></span>
              {modelStatus.model_type} Model Active
            </div>
          )}
        </div>
        
        <form onSubmit={handleSubmit} className="input-form">
          <div className="input-group">
            <label htmlFor="text-input">Enter your text</label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type or paste your text here... (e.g., 'I love this product!' or 'This service is terrible.')"
              rows={4}
              maxLength={1000}
            />
            <div className="char-count">{text.length}/1000</div>
          </div>
          
          <button 
            type="submit" 
            disabled={loading || !text.trim()}
            className={`analyze-btn ${loading ? 'loading' : ''}`}
          >
            {loading ? (
              <>
                <div className="spinner"></div>
                Analyzing...
              </>
            ) : (
              <>
                <span>🔍</span>
                Analyze Sentiment
              </>
            )}
          </button>
        </form>

        {result && (
          <div className="result-container">
            {result.error ? (
              <div className="error">
                <span>⚠️</span>
                {result.error}
              </div>
            ) : (
              <div className={`sentiment-result ${result.label}`}>
                <div className="sentiment-header">
                  <div className="sentiment-icon">
                    {getSentimentIcon(result.label)}
                  </div>
                  <div className="sentiment-info">
                    <h3>{result.label.charAt(0).toUpperCase() + result.label.slice(1)} Sentiment</h3>
                    <div className="confidence-score">
                      Confidence: 
                      <span 
                        className="score-value"
                        style={{ color: getConfidenceColor(result.score) }}
                      >
                        {(result.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="confidence-bar">
                  <div 
                    className={`confidence-fill ${result.label}`}
                    style={{ width: `${result.score * 100}%` }}
                  ></div>
                </div>
                
                <div className="sentiment-description">
                  {result.label === 'positive' 
                    ? "This text expresses positive emotions, satisfaction, or approval."
                    : "This text expresses negative emotions, dissatisfaction, or criticism."
                  }
                </div>
              </div>
            )}
          </div>
        )}

        <div className="footer">
          <p>Powered by advanced NLP models • Real-time analysis</p>
        </div>
      </div>
    </div>
  );
}

export default App;
