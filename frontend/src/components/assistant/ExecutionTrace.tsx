import { Fragment } from 'react';
import type { EvidenceObject, JobExecutionTrace, JobResponse } from '../../lib/jobResponse';

interface ExecutionTraceProps {
  trace: JobExecutionTrace;
  response?: JobResponse;
  /** Controlled so an evidence chip can open the trace. */
  open: boolean;
  onToggle: () => void;
  selectedEvidenceId: string | null;
  onSelectEvidence: (id: string | null) => void;
}

const PLANNER_LABEL: Record<string, string> = {
  qwen3: 'Qwen3',
  registry_rules: 'Tool Registry rules (fallback)',
  template: 'evidence-only template (no LLM)',
  fixture: 'test fixture',
};

function time(ts: string | undefined): string {
  if (!ts) return '';
  const d = new Date(ts.endsWith('Z') || ts.includes('+') ? ts : `${ts}Z`);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleTimeString();
}

function text(value: unknown): string {
  if (value == null) return '';
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function regionLabel(ev: EvidenceObject): string {
  if (!ev.spatial_region) return 'no spatial region';
  if (ev.processing_parameters?.spatial_region_is_full_image) return `${ev.spatial_region.type} · whole image footprint`;
  return `${ev.spatial_region.type} · localized region`;
}

/** Renders the stored execution trace of one job exactly as the backend recorded it. */
export default function ExecutionTrace({ trace, response, open, onToggle, selectedEvidenceId, onSelectEvidence }: ExecutionTraceProps) {

  const plannedEntry = [...trace.trace].reverse().find((e) => e.task_spec);
  const taskSpec = plannedEntry?.task_spec as Record<string, unknown> | undefined;
  const selection = trace.trace.filter((e) => e.stage === 'TOOL_SELECTION');
  const toolEvents = trace.execution_trace.filter((e) => e.event_type !== 'workflow_start' || e.status === 'failed');
  const verification = trace.verification_result;
  const rejectedIds = new Set((verification?.rejected_claims ?? []).map((c) => c.evidence_id));
  const evidence = response?.evidence_objects ?? [];

  return (
    <div className="exec-trace">
      <button className="exec-trace__toggle" onClick={onToggle} aria-expanded={open}>
        <span className="exec-trace__node" aria-hidden="true" />
        <span className="exec-trace__toggle-label">{open ? 'Hide execution trace' : 'Show execution trace'}</span>
      </button>

      {open && (
        <div className="exec-trace__body rail rail--trace">
          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">01</span>Planned task</h4>
            {taskSpec ? (
              <dl className="exec-trace__kv">
                <dt>primary_task</dt><dd>{text(taskSpec.primary_task)}</dd>
                <dt>required_modalities</dt><dd>{text(taskSpec.required_modalities)}</dd>
                <dt>temporal_requirement</dt><dd>{text(taskSpec.temporal_requirement)}</dd>
              </dl>
            ) : (
              <p className="exec-trace__empty">The job stopped before a task was planned.</p>
            )}
            {trace.planner && (
              <dl className="exec-trace__kv">
                {Object.entries(trace.planner).map(([k, v]) => (
                  <Fragment key={k}><dt>{k}</dt><dd>{v ? (PLANNER_LABEL[v] ?? v) : '—'}</dd></Fragment>
                ))}
              </dl>
            )}
            {selection.map((entry, i) => (
              <div key={i} className="exec-trace__line">
                <span className="exec-trace__time">{time(entry.timestamp)}</span> {entry.message}
                {Array.isArray(entry.details) && entry.details.length > 0 && (
                  <ul className="exec-trace__list">{entry.details.map((d, j) => <li key={j}>{text(d)}</li>)}</ul>
                )}
              </div>
            ))}
          </section>

          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">02</span>Executed tools</h4>
            {toolEvents.length === 0 ? (
              <p className="exec-trace__empty">No specialist tool was executed.</p>
            ) : (
              <ul className="exec-trace__list">
                {toolEvents.map((e, i) => {
                  const summary = trace.results.find((r) => r.call_id === e.call_id);
                  const duration = e.status !== 'running' ? summary?.execution_metadata?.duration_ms : undefined;
                  return (
                    <li key={i} className={`exec-trace__event exec-trace__event--${e.status}`}>
                      <span className="exec-trace__time">{time(e.timestamp)}</span>{' '}
                      {e.event_type}{e.tool_id ? ` · ${e.tool_id}` : ''} · {e.status}
                      {duration != null ? ` · ${text(duration)} ms` : ''}
                      {(e.error || e.reason) && <div className="exec-trace__detail">{text(e.error ?? e.reason)}</div>}
                    </li>
                  );
                })}
              </ul>
            )}
            {trace.replan_history.length > 0 && (
              <p className="exec-trace__detail">Recoveries: {trace.replan_history.length}</p>
            )}
          </section>

          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">03</span>Verification</h4>
            {verification ? (
              <>
                <dl className="exec-trace__kv">
                  <dt>status</dt><dd>{verification.status}</dd>
                  <dt>rejected_claims</dt><dd>{verification.rejected_claims?.length ?? 0}</dd>
                </dl>
                {(verification.triggers_fired ?? []).length > 0 && (
                  <ul className="exec-trace__list">
                    {(verification.triggers_fired ?? []).map((t, i) => <li key={i}>{text(t)}</li>)}
                  </ul>
                )}
              </>
            ) : (
              <p className="exec-trace__empty">Verification did not run for this job.</p>
            )}
          </section>

          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">04</span>Evidence regions</h4>
            {evidence.length === 0 ? (
              <p className="exec-trace__empty">No evidence objects were produced.</p>
            ) : (
              <ul className="exec-trace__evidence">
                {evidence.map((ev) => {
                  const selected = selectedEvidenceId === ev.evidence_id;
                  return (
                    <li key={ev.evidence_id}>
                      <button
                        className={`evidence-card ${selected ? 'evidence-card--selected' : ''} ${rejectedIds.has(ev.evidence_id) ? 'evidence-card--rejected' : ''}`}
                        onClick={() => onSelectEvidence(selected ? null : ev.evidence_id)}
                        disabled={!ev.spatial_region}
                        title={ev.spatial_region ? 'Highlight on the map' : 'This evidence has no spatial region'}
                      >
                        <span className="evidence-card__claim">{rejectedIds.has(ev.evidence_id) && '[Rejected] '}{ev.claim}</span>
                        <span className="evidence-card__meta">
                          <span>{ev.evidence_id}</span>
                          <span>{ev.source_model}</span>
                          <span title={text(ev.processing_parameters?.confidence_source)}>model confidence (uncalibrated): {(ev.confidence * 100).toFixed(0)}%</span>
                          <span>{regionLabel(ev)}</span>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">05</span>Pipeline stages</h4>
            <ul className="exec-trace__list">
              {trace.trace.map((entry, i) => (
                <li key={i}>
                  <span className="exec-trace__time">{time(entry.timestamp)}</span> <strong>{entry.stage}</strong> — {entry.message}
                  {entry.error && <div className="exec-trace__detail">{entry.error}</div>}
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </div>
  );
}
