import React from 'react';

export default function ExplanationPanel({ topFeatures = [], explanationMethod = 'importance_x_deviation' }) {
  if (!topFeatures || topFeatures.length === 0) {
    return (
      <div className="explanation-panel">
        <h3 className="card-subheading">Features that stand out for this wallet</h3>
        <p className="text-muted">No explanatory features available for this address.</p>
      </div>
    );
  }

  // Calculate max absolute contribution to normalize bar widths
  const maxAbsContrib = Math.max(...topFeatures.map(f => Math.abs(f.contribution || 0)), 0.0001);

  return (
    <div className="explanation-panel">
      <div className="explanation-header">
        <h3 className="card-subheading">Features that stand out for this wallet</h3>
        <span className="method-tag">
          Method: <strong className="mono">{explanationMethod}</strong>
        </span>
      </div>

      <p className="explanation-caption">
        Feature ranking is based on {explanationMethod === 'shap' ? 'SHAP TreeExplainer values' : 'importance times deviation'}. Note: This highlights salient statistical deviations, not a causal reason for the score.
      </p>

      <div className="feature-bars-list">
        {topFeatures.map((feat, index) => {
          const rawContrib = feat.contribution || 0;
          const absContrib = Math.abs(rawContrib);
          const barWidthPercent = Math.min(100, Math.round((absContrib / maxAbsContrib) * 100));

          return (
            <div key={index} className="feature-bar-row">
              <div className="feature-info-col">
                <span className="feature-name" title={feat.name}>
                  {feat.name}
                </span>
                <span className="feature-raw-value">
                  Val: {typeof feat.value === 'number' ? feat.value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : feat.value}
                </span>
              </div>

              <div className="bar-track">
                {/* Neutral coloration: no red or green allowed */}
                <div
                  className="bar-fill neutral-fill"
                  style={{ width: `${barWidthPercent}%` }}
                />
              </div>

              <div className="feature-contrib-col">
                <span className="contrib-value mono">
                  {rawContrib > 0 ? `+${rawContrib.toFixed(4)}` : rawContrib.toFixed(4)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
