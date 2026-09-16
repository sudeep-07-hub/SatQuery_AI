import { useState } from 'react';
import BeforeAfterViewer from './BeforeAfterViewer';

interface ResultPanelProps {
  apiBase: string;
  status: string;
  result: any;
  evidenceGraph: any;
  jobId: string;
}

const STOPPED_COPY: Record<string, { title: string; desc: string }> = {
  PRECONDITION_FAILED: {
    title: 'Inputs Rejected',
    desc: 'The uploaded images did not pass input qualification (format, readability or spatial compatibility).',
  },
  INSUFFICIENT_OBSERVATIONS: {
    title: 'Inputs Do Not Fit the Request',
    desc: 'The images uploaded cannot satisfy what the query needs (for example, change detection needs two images of the same area).',
  },
  ABSTAIN: {
    title: 'System Abstained',
    desc: 'No available engine can answer this request reliably, so no answer was produced.',
  },
  INSUFFICIENT_EVIDENCE: {
    title: 'Insufficient Evidence',
    desc: 'Verification rejected the evidence, so no answer is given.',
  },
  MODEL_UNAVAILABLE: {
    title: 'Model Unavailable',
    desc: 'A required model could not be loaded in this environment.',
  },
  FAILED: {
    title: 'Analysis Failed',
    desc: 'An internal error stopped the pipeline.',
  },
};

const PLANNER_LABEL: Record<string, string> = {
  qwen3: 'Qwen3',
  registry_rules: 'Registry rules',
  template: 'Evidence template',
  fixture: 'Fixture',
};

function ExportLinks({ apiBase, jobId, full, hasMask = false }: { apiBase: string; jobId: string; full: boolean; hasMask?: boolean }) {
  const links = full
    ? [['pdf', '📄 PDF Report'], ['geojson', '🗺️ GeoJSON'], ['json', '🔍 JSON Trace'], ...(hasMask ? [['png', '🖼️ Change Mask PNG']] : [])]
    : [['json', '🔍 JSON Trace']];
  return (
    <div className="exports-section">
      <h3>Export Artifacts</h3>
      <div className="export-links">
        {links.map(([fmt, label]) => (
          <a key={fmt} href={`${apiBase}/api/jobs/${jobId}/export/${fmt}`} className="export-link">{label}</a>
        ))}
      </div>
    </div>
  );
}

export default function ResultPanel({ apiBase, status, result, evidenceGraph, jobId }: ResultPanelProps) {
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);

  if (status !== 'DONE') {
    const copy = STOPPED_COPY[status] || { title: status, desc: '' };
    const reasons: string[] = (result?.failed || []).map((f: any) => f.reason || f.check).filter(Boolean);
    return (
      <div className={`result-panel result-panel--error result-panel--${status.toLowerCase()}`}>
        <h2>{copy.title}</h2>
        <div className="error-status">Status: {status}</div>
        <p className="error-description">{copy.desc}</p>
        {result?.final_answer && <div className="final-answer">{result.final_answer}</div>}
        {result?.caveats && result.caveats.length > 0 && (
          <div className="caveats-section">
            <strong>Caveats</strong>
            <ul>{result.caveats.map((c: string, idx: number) => <li key={idx}>{c}</li>)}</ul>
          </div>
        )}
        {reasons.length > 0 && (
          <div className="error-details">
            <strong>Specific reasons:</strong>
            <ul>{reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
          </div>
        )}
        <ExportLinks apiBase={apiBase} jobId={jobId} full={false} />
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-panel result-panel--processing">
        <div className="spinner"></div>
        <p>Loading result…</p>
      </div>
    );
  }

  const confidence = result.confidence || 0;
  const evidenceNodes = (evidenceGraph?.nodes || []).filter((n: any) => n.type === 'evidence');
  const rejectedIds = new Set((result.verification_result?.rejected_claims || []).map((c: any) => c.evidence_id));
  const planner = result.planner || {};
  const stats = result.change_statistics;

  return (
    <div className="result-panel">
      <div className="result-header">
        <h2>Analysis Result</h2>
        <div className="confidence-badge" title={result.confidence_note}>
          <span className="conf-label">Top evidence score {(confidence * 100).toFixed(0)}%</span>
          <span className="conf-value">uncalibrated · per-item sources below</span>
        </div>
      </div>

      <div className="planner-badges">
        {planner.llm_backend && <span className="pill">LLM: {planner.llm_backend}</span>}
        {planner.query_intelligence && <span className="pill">Intent: {PLANNER_LABEL[planner.query_intelligence] || planner.query_intelligence}</span>}
        {planner.tool_selection && <span className="pill">Tool choice: {PLANNER_LABEL[planner.tool_selection] || planner.tool_selection}</span>}
        {planner.answer && <span className="pill">Answer: {PLANNER_LABEL[planner.answer] || planner.answer}</span>}
      </div>

      <div className="final-answer">{result.final_answer}</div>

      {stats && (
        <div className="stat-row">
          <div className="stat"><span className="stat__value">{stats.changed_pixel_pct.toFixed(2)}%</span><span className="stat__label">of scene changed</span></div>
          {stats.changed_area_m2 != null && (
            <div className="stat"><span className="stat__value">{Math.round(stats.changed_area_m2).toLocaleString()} m²</span><span className="stat__label">changed area</span></div>
          )}
          <div className="stat"><span className="stat__value">{stats.region_count}</span><span className="stat__label">change regions</span></div>
          <div className="stat"><span className="stat__value">{stats.threshold} {stats.threshold_unit}</span><span className="stat__label">threshold</span></div>
        </div>
      )}

      {result.caveats && result.caveats.length > 0 && (
        <div className="caveats-section">
          <strong>Caveats</strong>
          <ul>{result.caveats.map((c: string, idx: number) => <li key={idx}>{c}</li>)}</ul>
        </div>
      )}

      <div className="claims-section">
        <h3>Evidence ({evidenceNodes.length}) · click to highlight</h3>
        <div className="evidence-list">
          {evidenceNodes.map((node: any) => {
            const ev = node.data;
            const isSelected = selectedEvidenceId === ev.evidence_id;
            const isRejected = rejectedIds.has(ev.evidence_id);
            return (
              <div
                key={ev.evidence_id}
                onClick={() => setSelectedEvidenceId(isSelected ? null : ev.evidence_id)}
                className={`evidence-card ${isSelected ? 'evidence-card--selected' : ''} ${isRejected ? 'evidence-card--rejected' : ''}`}
              >
                <div className="evidence-card__claim">{isRejected && '[Rejected] '}{ev.claim}</div>
                <div className="evidence-card__meta">
                  <span><strong>Model:</strong> {ev.source_model}</span>
                  <span><strong>Modality:</strong> {ev.modality}</span>
                  <span title={ev.processing_parameters?.confidence_source}><strong>Conf:</strong> {(ev.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <BeforeAfterViewer apiBase={apiBase} result={result} evidenceGraph={evidenceGraph} selectedEvidenceId={selectedEvidenceId} />

      <ExportLinks apiBase={apiBase} jobId={jobId} full={true} hasMask={Boolean(result.change_overlay)} />
    </div>
  );
}
