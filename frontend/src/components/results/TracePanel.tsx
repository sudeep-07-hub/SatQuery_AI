export interface TraceEntry {
  message: string;
  stage: string;
  timestamp: string;
  details?: unknown;
  error?: string;
  reason?: string;
  failed?: unknown;
  triggers?: unknown;
}

interface TracePanelProps {
  trace: TraceEntry[];
}

function asText(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
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

            {Array.isArray(entry.details) && entry.details.length > 0 && (
              <div className="trace-details">
                {entry.details.map((d, i) => (
                  <div key={i} className="trace-detail-item">• {asText(d)}</div>
                ))}
              </div>
            )}

            {entry.error && <div className="trace-reason">Error: {entry.error}</div>}
            {entry.reason && <div className="trace-reason">Reason: {entry.reason}</div>}
            {entry.triggers != null && <div className="trace-triggers">Triggers: {asText(entry.triggers)}</div>}
          </li>
        ))}
      </ul>
    </div>
  );
}
