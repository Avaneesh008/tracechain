import React from 'react';

export default function RiskGauge({ score = 0, label = 'licit' }) {
  // Clamped score between 0 and 1
  const clampedScore = Math.max(0, Math.min(1, score));
  const isIllicit = label === 'illicit' || clampedScore >= 0.5;

  // Semi-circle gauge geometry
  const radius = 80;
  const strokeWidth = 14;
  const center = 100;
  const circumference = Math.PI * radius; // Half-circle perimeter
  const strokeDashoffset = circumference * (1 - clampedScore);

  // Gradient / theme color based on severity for gauge stroke
  const strokeColor = clampedScore >= 0.7 ? '#dc2626' : clampedScore >= 0.4 ? '#f59e0b' : '#10b981';

  return (
    <div className="risk-gauge-container">
      <div className="gauge-svg-wrap">
        <svg viewBox="0 0 200 125" className="gauge-svg">
          {/* Background track */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#e2e8f0"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Active progress arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{ transition: 'stroke-dashoffset 0.8s ease-out' }}
          />
          {/* Score text */}
          <text
            x={center}
            y={85}
            textAnchor="middle"
            className="gauge-score-text"
          >
            {clampedScore.toFixed(4)}
          </text>
          <text
            x={center}
            y={105}
            textAnchor="middle"
            className="gauge-label-text"
          >
            Risk Score
          </text>
        </svg>
      </div>

      <div className="gauge-meta">
        <div className="risk-badge-row">
          <span className={`risk-classification-badge ${isIllicit ? 'illicit' : 'licit'}`}>
            Classification: {label ? label.toUpperCase() : (isIllicit ? 'ILLICIT' : 'LICIT')}
          </span>
        </div>
        <p className="triage-disclaimer">
          Note: This value is a <strong>risk score</strong>, not an absolute probability. Scores are intended for investigation triage only and never for automatic action.
        </p>
      </div>
    </div>
  );
}
