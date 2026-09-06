import React from 'react';

export interface TraceEntry {
  message: string;
  stage: string;
  timestamp: string;
  details?: string[];
  task_spec?: any;
  reason?: string;
  failed?: any;
  triggers?: string[];
}

interface TracePanelProps {
  trace: TraceEntry[];
}

export default function TracePanel({ trace }: TracePanelProps) {
  if (!trace || trace.length === 0) {
    return <div className="trace-panel-empty">No trace data yet...</div>;
  }

  return (
    <div className="trace-panel">
      <h3>Execution Trace</h3>
      <ul className="trace-list">
        {trace.map((entry, idx) => (
          <li key={idx} className={`trace-item trace-item--${entry.stage.toLowerCase()}`}>
            <div className="trace-item-header">
              <span className="trace-stage">{entry.stage}</span>
              <span className="trace-time">{new Date(entry.timestamp).toLocaleTimeString()}</span>
            </div>
            <div className="trace-message">{entry.message}</div>
            
            {entry.details && (
              <div className="trace-details">
                {entry.details.map((d, i) => (
                  <div key={i} className="trace-detail-item">• {d}</div>
                ))}
              </div>
            )}
            
            {entry.reason && (
              <div className="trace-reason">Reason: {entry.reason}</div>
            )}
            
            {entry.triggers && (
              <div className="trace-triggers">
                Triggers: {entry.triggers.join(', ')}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
