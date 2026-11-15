import sampleWalletsData from './mock/sample_wallets.json';
import scoreIllicitData from './mock/score_illicit.json';
import scoreLicitData from './mock/score_licit.json';
import traceIllicitData from './mock/trace_illicit.json';
import traceLicitData from './mock/trace_licit.json';
import metricsData from './mock/metrics.json';

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function handleResponse(res) {
  if (!res.ok) {
    let detail = '';
    try {
      const errJson = await res.json();
      detail = errJson.detail || '';
    } catch {
      detail = res.statusText;
    }
    const err = new Error(detail || `Request failed with status ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function fetchHealth() {
  if (USE_MOCK) {
    return { status: 'ok', model_loaded: true, graph_nodes: 822942 };
  }
  const res = await fetch(`${API_BASE}/api/health`);
  return handleResponse(res);
}

export async function fetchSampleWallets(label = 'illicit', n = 5) {
  if (USE_MOCK) {
    const list = sampleWalletsData[label]?.wallets || [];
    return { wallets: list.slice(0, n) };
  }
  const res = await fetch(`${API_BASE}/api/wallets/sample?label=${encodeURIComponent(label)}&n=${n}`);
  return handleResponse(res);
}

export async function fetchScore(address) {
  if (USE_MOCK) {
    const trimmed = address.trim();
    if (!trimmed) {
      const err = new Error('Address is required');
      err.status = 400;
      throw err;
    }
    if (trimmed === 'invalid' || trimmed.toLowerCase().includes('unknown')) {
      const err = new Error(`Address '${trimmed}' not found in dataset.`);
      err.status = 404;
      throw err;
    }
    // Return illicit mock if address contains illicit or starts with 12c, else licit mock
    if (trimmed.toLowerCase().includes('illicit') || trimmed.startsWith('12c') || trimmed.startsWith('15x')) {
      return { ...scoreIllicitData, address: trimmed };
    }
    return { ...scoreLicitData, address: trimmed };
  }
  const res = await fetch(`${API_BASE}/api/score/${encodeURIComponent(address.trim())}`);
  return handleResponse(res);
}

export async function fetchTrace(address) {
  if (USE_MOCK) {
    const trimmed = address.trim();
    if (trimmed.toLowerCase().includes('illicit') || trimmed.startsWith('12c') || trimmed.startsWith('15x')) {
      return traceIllicitData;
    }
    return traceLicitData;
  }
  const res = await fetch(`${API_BASE}/api/trace/${encodeURIComponent(address.trim())}`);
  return handleResponse(res);
}

export async function fetchMetrics() {
  if (USE_MOCK) {
    return metricsData;
  }
  const res = await fetch(`${API_BASE}/api/metrics`);
  return handleResponse(res);
}
