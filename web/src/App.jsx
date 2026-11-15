import React, { useState, useEffect } from 'react';
import { fetchHealth, fetchSampleWallets, fetchScore, fetchTrace } from './api';
import RiskGauge from './components/RiskGauge';
import VaspBadge from './components/VaspBadge';
import ExplanationPanel from './components/ExplanationPanel';
import FundFlowGraph from './components/FundFlowGraph';
import MetricsView from './components/MetricsView';

export default function App() {
  const [activeTab, setActiveTab] = useState('investigate'); // 'investigate' | 'metrics'
  const [addressInput, setAddressInput] = useState('');
  const [analyzedAddress, setAnalyzedAddress] = useState('');
  
  const [scoreData, setScoreData] = useState(null);
  const [traceData, setTraceData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [samplesLoading, setSamplesLoading] = useState(false);
  const [apiHealth, setApiHealth] = useState(null);

  // Check API health and check URL query params on load
  useEffect(() => {
    fetchHealth()
      .then(data => setApiHealth(data))
      .catch(() => setApiHealth({ status: 'offline', model_loaded: false, graph_nodes: 0 }));

    const params = new URLSearchParams(window.location.search);
    const tabParam = params.get('tab');
    if (tabParam === 'metrics') {
      setActiveTab('metrics');
    }
    const addrParam = params.get('address');
    if (addrParam) {
      setAddressInput(addrParam);
      handleAnalyze(addrParam);
    }
  }, []);

  const handleAnalyze = async (addrToAnalyze) => {
    const target = (addrToAnalyze || addressInput).trim();
    if (!target) return;

    setLoading(true);
    setError(null);
    setAnalyzedAddress(target);

    try {
      const [scoreRes, traceRes] = await Promise.all([
        fetchScore(target),
        fetchTrace(target)
      ]);
      setScoreData(scoreRes);
      setTraceData(traceRes);
      const params = new URLSearchParams(window.location.search);
      if (params.get('scroll') === 'graph') {
        setTimeout(() => {
          const el = document.getElementById('fund-flow-graph');
          if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
        }, 300);
      }
    } catch (err) {
      setScoreData(null);
      setTraceData(null);
      if (err.status === 404) {
        setError(`Address '${target}' was not found in the dataset (HTTP 404). Only addresses present in the Elliptic++ Actors dataset can be scored.`);
      } else if (err.status === 503) {
        setError(`Service Unavailable (HTTP 503): Backend model or graph artifacts are currently missing or unloaded.`);
      } else {
        setError(err.message || 'An error occurred while contacting the TraceChain API.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClick = async (label) => {
    setSamplesLoading(true);
    setError(null);
    try {
      const data = await fetchSampleWallets(label, 3);
      if (data.wallets && data.wallets.length > 0) {
        const sampleAddr = data.wallets[0].address;
        setAddressInput(sampleAddr);
        handleAnalyze(sampleAddr);
      }
    } catch (err) {
      setError(`Could not fetch ${label} sample wallet: ${err.message}`);
    } finally {
      setSamplesLoading(false);
    }
  };

  const handleClear = () => {
    setAddressInput('');
    setAnalyzedAddress('');
    setScoreData(null);
    setTraceData(null);
    setError(null);
  };

  return (
    <div className="app-container">
      {/* Header / Navbar */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-logo-badge">TC</div>
          <div>
            <h1 className="brand-title">TraceChain</h1>
            <p className="brand-tagline">
              Bitcoin Illicit-Risk Scoring & VASP-Like Hub Attribution
            </p>
          </div>
        </div>

        <div className="header-nav">
          <div className="nav-tabs">
            <button
              className={`nav-tab-btn ${activeTab === 'investigate' ? 'active' : ''}`}
              onClick={() => setActiveTab('investigate')}
            >
              Investigate Wallet
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
              onClick={() => setActiveTab('metrics')}
            >
              Metrics & Attribution
            </button>
          </div>

          <div className="api-status-badge">
            <span className={`status-indicator ${apiHealth?.model_loaded ? 'online' : 'warn'}`} />
            <span className="status-text">
              {apiHealth?.model_loaded ? 'Model Loaded' : 'API Connecting'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="app-main">
        {activeTab === 'investigate' ? (
          <div className="investigate-view">
            {/* Search Box & Controls */}
            <section className="search-section card">
              <form
                className="search-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  handleAnalyze();
                }}
              >
                <div className="search-input-wrap">
                  <span className="search-icon">&#128269;</span>
                  <input
                    type="text"
                    className="search-input mono"
                    placeholder="Enter Bitcoin address (e.g. 12cGPFdJHGxZ9VZrV16gJWvv5228DnnK)..."
                    value={addressInput}
                    onChange={(e) => setAddressInput(e.target.value)}
                  />
                  {addressInput && (
                    <button
                      type="button"
                      className="btn-clear"
                      onClick={handleClear}
                      title="Clear address"
                    >
                      &times;
                    </button>
                  )}
                </div>

                <div className="search-actions">
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={loading || !addressInput.trim()}
                  >
                    {loading ? 'Analyzing...' : 'Score & Trace'}
                  </button>
                </div>
              </form>

              {/* Sample Buttons */}
              <div className="sample-buttons-row">
                <span className="sample-label">Try a sample wallet:</span>
                <button
                  type="button"
                  className="btn-sample illicit"
                  onClick={() => handleSampleClick('illicit')}
                  disabled={samplesLoading || loading}
                >
                  Sample Illicit Wallet
                </button>
                <button
                  type="button"
                  className="btn-sample licit"
                  onClick={() => handleSampleClick('licit')}
                  disabled={samplesLoading || loading}
                >
                  Sample Licit Wallet
                </button>
              </div>
            </section>

            {/* Error Message Display */}
            {error && (
              <div className="error-alert">
                <div className="error-icon">&#9888;</div>
                <div className="error-content">
                  <h4>Investigation Error</h4>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {/* Loading Indicator */}
            {loading && (
              <div className="loading-card card">
                <div className="spinner" />
                <p>Computing risk score, feature explanation, and shortest-path trace...</p>
              </div>
            )}

            {/* Analysis Results View */}
            {!loading && scoreData && (
              <div className="results-grid">
                {/* Top Row: Risk Gauge & VASP Attribution Card */}
                <div className="results-top-row">
                  <div className="card gauge-card-box">
                    <div className="card-header-bar">
                      <span className="card-title">Risk Assessment</span>
                      <span className="address-pill mono" title={scoreData.address}>
                        {scoreData.address}
                      </span>
                    </div>
                    <RiskGauge
                      score={scoreData.risk_score}
                      label={scoreData.risk_label}
                    />
                  </div>

                  <div className="card vasp-card-box">
                    <VaspBadge nearestVasp={scoreData.nearest_vasp} />
                  </div>
                </div>

                {/* Explanation Panel */}
                <div className="card explanation-card-box">
                  <ExplanationPanel
                    topFeatures={scoreData.top_features}
                    explanationMethod={scoreData.explanation_method}
                  />
                </div>

                {/* Fund-flow Graph */}
                <div id="fund-flow-graph" className="card graph-card-box">
                  <FundFlowGraph
                    traceData={traceData}
                    targetAddress={analyzedAddress}
                  />
                </div>
              </div>
            )}

            {/* Empty State / Welcome Guide */}
            {!loading && !scoreData && !error && (
              <div className="empty-state-guide card">
                <div className="guide-icon">&#128737;</div>
                <h3>TraceChain Bitcoin Analysis Dashboard</h3>
                <p>
                  Search for a Bitcoin address from the Elliptic++ dataset or click one of the sample buttons above to run illicit-risk scoring, feature explanation, and nearest VASP shortest-path attribution.
                </p>
                <div className="guide-features-list">
                  <div className="guide-feature-item">
                    <strong>Risk Scoring:</strong> Graph-enhanced Random Forest (F1 0.5797 vs baseline 0.4292).
                  </div>
                  <div className="guide-feature-item">
                    <strong>VASP Attribution:</strong> Multi-source BFS shortest path to exchange proxy seeds (K=30).
                  </div>
                  <div className="guide-feature-item">
                    <strong>Honest Explanations:</strong> Importance &times; deviation statistical ranking with method transparency.
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="metrics-tab-view">
            <MetricsView />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-content">
          <span>TraceChain v2 &bull; Proof-of-concept on Elliptic++ Actors dataset</span>
          <span className="footer-scope-note">
            VASP seeds are proxies based on network structure. Risk scores are for triage only.
          </span>
        </div>
      </footer>
    </div>
  );
}
