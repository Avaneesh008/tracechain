import React, { useRef, useEffect, useState, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const NODE_COLORS = {
  wallet: '#2563eb', // Vibrant Blue
  path: '#f59e0b',   // Amber / Orange
  vasp: '#7c3aed',   // Royal Purple
  neighbor: '#94a3b8' // Slate Muted
};

const NODE_RADIUS = {
  wallet: 8,
  vasp: 9,
  path: 6,
  neighbor: 4
};

export default function FundFlowGraph({ traceData, targetAddress }) {
  const containerRef = useRef(null);
  const fgRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 700, height: 420 });

  // Update container width on resize
  useEffect(() => {
    function updateDimensions() {
      if (containerRef.current) {
        const { clientWidth } = containerRef.current;
        setDimensions({
          width: Math.max(320, clientWidth - 24),
          height: 440,
        });
      }
    }

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const pathSet = useMemo(() => {
    return new Set(traceData?.path || []);
  }, [traceData]);

  // Prepare graph data
  const graphData = useMemo(() => {
    if (!traceData || !traceData.nodes || traceData.nodes.length === 0) {
      return { nodes: [], links: [] };
    }

    const nodes = traceData.nodes.map(n => ({
      id: n.id,
      kind: n.kind || 'neighbor',
      isPath: pathSet.has(n.id),
      isTarget: n.id === targetAddress,
      color: NODE_COLORS[n.kind] || NODE_COLORS.neighbor,
      val: NODE_RADIUS[n.kind] || 4
    }));

    const links = (traceData.edges || []).map(e => {
      const isPathEdge = pathSet.has(e.source) && pathSet.has(e.target);
      return {
        source: e.source,
        target: e.target,
        isPathEdge,
        color: isPathEdge ? '#f59e0b' : '#cbd5e1',
        width: isPathEdge ? 2.5 : 1
      };
    });

    return { nodes, links };
  }, [traceData, targetAddress, pathSet]);

  // Center graph after rendering
  useEffect(() => {
    if (fgRef.current && graphData.nodes.length > 0) {
      setTimeout(() => {
        fgRef.current.zoomToFit(400, 30);
      }, 300);
    }
  }, [graphData]);

  const handleResetZoom = () => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400, 30);
    }
  };

  if (!traceData || !traceData.nodes || traceData.nodes.length === 0) {
    return (
      <div className="graph-card-empty" ref={containerRef}>
        <h3 className="card-subheading">Fund-Flow Subgraph</h3>
        <p className="text-muted">No graph trace available for this address.</p>
      </div>
    );
  }

  return (
    <div className="fund-flow-graph-container" ref={containerRef}>
      <div className="graph-header-row">
        <div>
          <h3 className="card-subheading">Fund-Flow Trace & Neighbor Subgraph</h3>
          <p className="graph-meta-text">
            Shortest path to nearest VASP seed with up to 12 immediate neighbors ({graphData.nodes.length} nodes, {graphData.links.length} edges)
          </p>
        </div>

        <div className="graph-controls">
          <button type="button" className="btn-small-outline" onClick={handleResetZoom}>
            Reset View
          </button>
        </div>
      </div>

      {/* Graph Legend */}
      <div className="graph-legend">
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: NODE_COLORS.wallet }} />
          <span>Queried Wallet</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: NODE_COLORS.path }} />
          <span>Shortest Path Hop</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: NODE_COLORS.vasp }} />
          <span>VASP Seed</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: NODE_COLORS.neighbor }} />
          <span>Neighbor</span>
        </div>
        <div className="legend-item">
          <span className="legend-line" style={{ backgroundColor: '#f59e0b' }} />
          <span>Path to VASP</span>
        </div>
      </div>

      {/* Force Graph View */}
      <div className="graph-canvas-frame">
        <ForceGraph2D
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphData}
          nodeLabel={node => `${node.id}\nKind: ${node.kind.toUpperCase()}`}
          nodeColor={node => node.color}
          nodeVal={node => node.val}
          linkColor={link => link.color}
          linkWidth={link => link.width}
          linkDirectionalArrowLength={link => (link.isPathEdge ? 4 : 0)}
          linkDirectionalArrowRelPos={0.9}
          d3VelocityDecay={0.4}
          cooldownTicks={100}
        />
      </div>
    </div>
  );
}
