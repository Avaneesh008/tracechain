import React, { useEffect, useState } from 'react';
import { fetchMetrics } from '../api';

export default function MetricsView() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetchMetrics()
      .then(data => {
        if (mounted) {
          setMetrics(data);
          setError(null);
          setLoading(false);
        }
      })
      .catch(err => {
        if (mounted) {
          setError(err.message || 'Failed to load model metrics from API');
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="metrics-loading-state">
        <div className="spinner" />
        <p>Loading model metrics and attribution statistics from API...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="metrics-error-state">
        <div className="error-banner">
          <h4>Failed to load metrics</h4>
          <p>{error}</p>
          <button className="btn-small-outline" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!metrics) {
    return null;
  }

  const baselineAll = metrics.baseline?.test_all;
  const baselineCold = metrics.baseline?.test_cold;
  const graphAll = metrics.graph_enhanced?.test_all;
  const graphCold = metrics.graph_enhanced?.test_cold;
  const attrStats = metrics.attribution_stats?.active_index_stats;
  const sensitivity = metrics.attribution_stats?.sensitivity_by_k;

  return (
    <div className="metrics-view-container">
      <div className="metrics-header">
        <h2 className="section-title">Model Evaluation & Attribution Performance</h2>
        <p className="section-subtitle">
          Real metrics loaded live from <span className="mono">/api/metrics</span>. All evaluations reflect the temporal test split (time steps 35 to 49) at classification threshold 0.5.
        </p>
      </div>

      {/* Main Model Comparison Table */}
      <div className="metrics-card">
        <h3 className="card-subheading">Model Performance Comparison (Illicit Class)</h3>
        <p className="table-caption">
          Comparing the 55-feature Baseline model with the 60-feature Graph-Enhanced model across all test rows and cold test addresses (addresses never seen in training).
        </p>

        <div className="table-responsive">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Evaluation Metric</th>
                <th>Baseline (All Test)</th>
                <th>Baseline (Cold Test)</th>
                <th>Graph-Enhanced (All Test)</th>
                <th>Graph-Enhanced (Cold Test)</th>
                <th>All Test Delta</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Precision</strong></td>
                <td className="mono">{baselineAll ? baselineAll.precision.toFixed(4) : 'N/A'}</td>
                <td className="mono">{baselineCold ? baselineCold.precision.toFixed(4) : 'N/A'}</td>
                <td className="mono bold-highlight">{graphAll ? graphAll.precision.toFixed(4) : 'N/A'}</td>
                <td className="mono bold-highlight">{graphCold ? graphCold.precision.toFixed(4) : 'N/A'}</td>
                <td className="mono delta-pos">
                  {baselineAll && graphAll ? `+${(graphAll.precision - baselineAll.precision).toFixed(4)}` : 'N/A'}
                </td>
              </tr>
              <tr>
                <td><strong>Recall</strong></td>
                <td className="mono">{baselineAll ? baselineAll.recall.toFixed(4) : 'N/A'}</td>
                <td className="mono">{baselineCold ? baselineCold.recall.toFixed(4) : 'N/A'}</td>
                <td className="mono bold-highlight">{graphAll ? graphAll.recall.toFixed(4) : 'N/A'}</td>
                <td className="mono bold-highlight">{graphCold ? graphCold.recall.toFixed(4) : 'N/A'}</td>
                <td className="mono delta-pos">
                  {baselineAll && graphAll ? `+${(graphAll.recall - baselineAll.recall).toFixed(4)}` : 'N/A'}
                </td>
              </tr>
              <tr>
                <td><strong>F1 Score</strong></td>
                <td className="mono">{baselineAll ? baselineAll.f1.toFixed(4) : 'N/A'}</td>
                <td className="mono">{baselineCold ? baselineCold.f1.toFixed(4) : 'N/A'}</td>
                <td className="mono bold-highlight">{graphAll ? graphAll.f1.toFixed(4) : 'N/A'}</td>
                <td className="mono bold-highlight">{graphCold ? graphCold.f1.toFixed(4) : 'N/A'}</td>
                <td className="mono delta-pos">
                  {baselineAll && graphAll ? `+${(graphAll.f1 - baselineAll.f1).toFixed(4)}` : 'N/A'}
                </td>
              </tr>
              <tr>
                <td><strong>PR-AUC</strong></td>
                <td className="mono">{baselineAll ? baselineAll.pr_auc.toFixed(4) : 'N/A'}</td>
                <td className="mono">{baselineCold ? baselineCold.pr_auc.toFixed(4) : 'N/A'}</td>
                <td className="mono bold-highlight">{graphAll ? graphAll.pr_auc.toFixed(4) : 'N/A'}</td>
                <td className="mono bold-highlight">{graphCold ? graphCold.pr_auc.toFixed(4) : 'N/A'}</td>
                <td className="mono delta-pos">
                  {baselineAll && graphAll ? `+${(graphAll.pr_auc - baselineAll.pr_auc).toFixed(4)}` : 'N/A'}
                </td>
              </tr>
              <tr className="subrow">
                <td>Evaluated Rows</td>
                <td className="mono">{baselineAll ? baselineAll.rows.toLocaleString() : 'N/A'}</td>
                <td className="mono">{baselineCold ? baselineCold.rows.toLocaleString() : 'N/A'}</td>
                <td className="mono">{graphAll ? graphAll.rows.toLocaleString() : 'N/A'}</td>
                <td className="mono">{graphCold ? graphCold.rows.toLocaleString() : 'N/A'}</td>
                <td className="mono text-muted">-</td>
              </tr>
              <tr className="subrow">
                <td>Illicit Labels Count</td>
                <td className="mono">{baselineAll ? baselineAll.illicit_count.toLocaleString() : 'N/A'}</td>
                <td className="mono">{baselineCold ? baselineCold.illicit_count.toLocaleString() : 'N/A'}</td>
                <td className="mono">{graphAll ? graphAll.illicit_count.toLocaleString() : 'N/A'}</td>
                <td className="mono">{graphCold ? graphCold.illicit_count.toLocaleString() : 'N/A'}</td>
                <td className="mono text-muted">-</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix Section */}
      <div className="metrics-card">
        <h3 className="card-subheading">Confusion Matrix Breakdown (All Test Rows: 97,756)</h3>
        <div className="cm-grid">
          <div className="cm-card">
            <h4>Baseline Model (55 features)</h4>
            {baselineAll?.confusion_matrix ? (
              <div className="cm-matrix">
                <div className="cm-cell"><span className="cm-label">True Positives (TP)</span><span className="cm-val mono">{baselineAll.confusion_matrix.tp.toLocaleString()}</span></div>
                <div className="cm-cell"><span className="cm-label">False Positives (FP)</span><span className="cm-val mono">{baselineAll.confusion_matrix.fp.toLocaleString()}</span></div>
                <div className="cm-cell"><span className="cm-label">False Negatives (FN)</span><span className="cm-val mono">{baselineAll.confusion_matrix.fn.toLocaleString()}</span></div>
                <div className="cm-cell"><span className="cm-label">True Negatives (TN)</span><span className="cm-val mono">{baselineAll.confusion_matrix.tn.toLocaleString()}</span></div>
              </div>
            ) : <p>N/A</p>}
          </div>

          <div className="cm-card highlight-card">
            <h4>Graph-Enhanced Model (60 features)</h4>
            {graphAll?.confusion_matrix ? (
              <div className="cm-matrix">
                <div className="cm-cell"><span className="cm-label">True Positives (TP)</span><span className="cm-val mono bold-highlight">{graphAll.confusion_matrix.tp.toLocaleString()}</span></div>
                <div className="cm-cell"><span className="cm-label">False Positives (FP)</span><span className="cm-val mono">{graphAll.confusion_matrix.fp.toLocaleString()}</span></div>
                <div className="cm-cell"><span className="cm-label">False Negatives (FN)</span><span className="cm-val mono">{graphAll.confusion_matrix.fn.toLocaleString()}</span></div>
                <div className="cm-cell"><span className="cm-label">True Negatives (TN)</span><span className="cm-val mono">{graphAll.confusion_matrix.tn.toLocaleString()}</span></div>
              </div>
            ) : <p>N/A</p>}
          </div>
        </div>
      </div>

      {/* VASP Reachability & Attribution Statistics */}
      {attrStats && (
        <div className="metrics-card">
          <h3 className="card-subheading">VASP Attribution & Reachability (K=30 Proxy Seeds)</h3>
          <div className="stats-kpi-grid">
            <div className="kpi-item">
              <span className="kpi-label">Reachable Addresses</span>
              <span className="kpi-value mono">
                {attrStats.reachable_addresses?.toLocaleString()} / {attrStats.total_addresses?.toLocaleString()}
              </span>
              <span className="kpi-sub">{(attrStats.reachable_share * 100).toFixed(2)}% of full graph</span>
            </div>
            <div className="kpi-item">
              <span className="kpi-label">Reachable &le; 2 Hops</span>
              <span className="kpi-value mono">{(attrStats.reachable_within_2_hops_share * 100).toFixed(2)}%</span>
              <span className="kpi-sub">{attrStats.reachable_within_2_hops_count?.toLocaleString()} nodes</span>
            </div>
            <div className="kpi-item">
              <span className="kpi-label">Reachable &le; 4 Hops</span>
              <span className="kpi-value mono">{(attrStats.reachable_within_4_hops_share * 100).toFixed(2)}%</span>
              <span className="kpi-sub">{attrStats.reachable_within_4_hops_count?.toLocaleString()} nodes</span>
            </div>
            <div className="kpi-item">
              <span className="kpi-label">Reachable &le; 6 Hops</span>
              <span className="kpi-value mono">{(attrStats.reachable_within_6_hops_share * 100).toFixed(2)}%</span>
              <span className="kpi-sub">{attrStats.reachable_within_6_hops_count?.toLocaleString()} nodes</span>
            </div>
            <div className="kpi-item">
              <span className="kpi-label">Median Hops (Illicit)</span>
              <span className="kpi-value mono">{attrStats.median_hops_illicit}</span>
              <span className="kpi-sub">Labelled illicit addresses</span>
            </div>
            <div className="kpi-item">
              <span className="kpi-label">Median Hops (Licit)</span>
              <span className="kpi-value mono">{attrStats.median_hops_licit}</span>
              <span className="kpi-sub">Labelled licit addresses</span>
            </div>
          </div>

          {/* Seed Sensitivity Table */}
          {sensitivity && (
            <div className="sensitivity-section">
              <h4 className="sub-title">Reachability Sensitivity by Number of Proxy Seeds (K)</h4>
              <div className="table-responsive">
                <table className="metrics-table">
                  <thead>
                    <tr>
                      <th>Seed Count (K)</th>
                      <th>Reachable &le; 2 Hops</th>
                      <th>Reachable &le; 4 Hops</th>
                      <th>Reachable &le; 6 Hops</th>
                      <th>Median Hops (Illicit)</th>
                      <th>Median Hops (Licit)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(sensitivity).map(([k, s]) => (
                      <tr key={k} className={k === '30' ? 'active-k-row' : ''}>
                        <td><strong>K = {k} {k === '30' ? '(Default)' : ''}</strong></td>
                        <td className="mono">{(s.reachable_within_2_hops_share * 100).toFixed(2)}% ({s.reachable_within_2_hops_count?.toLocaleString()})</td>
                        <td className="mono">{(s.reachable_within_4_hops_share * 100).toFixed(2)}% ({s.reachable_within_4_hops_count?.toLocaleString()})</td>
                        <td className="mono">{(s.reachable_within_6_hops_share * 100).toFixed(2)}% ({s.reachable_within_6_hops_count?.toLocaleString()})</td>
                        <td className="mono">{s.median_hops_illicit}</td>
                        <td className="mono">{s.median_hops_licit}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
