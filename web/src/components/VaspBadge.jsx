import React from 'react';

export default function VaspBadge({ nearestVasp }) {
  if (!nearestVasp) {
    return (
      <div className="vasp-card-empty">
        <p className="text-muted">No VASP attribution data available.</p>
      </div>
    );
  }

  const {
    id = 'N/A',
    name = 'N/A',
    kind = 'proxy',
    hops = null,
    confidence = 'unattributed',
    confidence_score = null,
  } = nearestVasp;

  const isProxy = kind === 'proxy';
  const isRegistry = kind === 'registry';

  return (
    <div className="vasp-attribution-card">
      <div className="vasp-header-row">
        <h3 className="card-subheading">Nearest VASP Attribution</h3>
        <div className="vasp-kind-badges">
          {isProxy && (
            <span className="badge-proxy" title="Exchange-like proxy hub detected by graph structure">
              PROXY
            </span>
          )}
          {isRegistry && (
            <span className="badge-registry" title="Known registry address">
              REGISTRY
            </span>
          )}
        </div>
      </div>

      <div className="vasp-details-grid">
        <div className="detail-item">
          <span className="detail-label">VASP Hub ID</span>
          <span className="detail-value mono">{id}</span>
        </div>
        <div className="detail-item">
          <span className="detail-label">Hub Name</span>
          <span className="detail-value">{name}</span>
        </div>
        <div className="detail-item">
          <span className="detail-label">Graph Distance</span>
          <span className="detail-value">
            {hops !== null ? `${hops} hop${hops === 1 ? '' : 's'}` : 'Unattributed (Disconnected)'}
          </span>
        </div>
        <div className="detail-item">
          <span className="detail-label">Attribution Confidence</span>
          <span className={`detail-value confidence-pill ${confidence}`}>
            {confidence ? confidence.toUpperCase() : 'UNKNOWN'}
            {confidence_score !== null && ` (${(confidence_score * 100).toFixed(1)}%)`}
          </span>
        </div>
      </div>

      <div className="vasp-proxy-note">
        {isProxy && (
          <p className="note-text">
            <strong>Proxy hub notice:</strong> This seed is an exchange-like address identified through network degree and transaction fan-in/fan-out structure, not a verified real-world exchange entity.
          </p>
        )}
        {isRegistry && (
          <p className="note-text">
            <strong>Registry notice:</strong> Matched directly from the known verified VASP registry database.
          </p>
        )}
      </div>
    </div>
  );
}
