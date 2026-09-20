import { Fragment } from 'react';
import type { EvidenceObject, JobExecutionTrace, JobResponse } from '../../lib/jobResponse';
import { useT } from '../../i18n/useT';
import { isLocaleCode, LOCALE_ENDONYM } from '../../i18n/types';
import type { Translate } from '../../i18n/I18nProvider';
import type { TranslationKey } from '../../i18n/types';

interface ExecutionTraceProps {
  trace: JobExecutionTrace;
  response?: JobResponse;
  /** Controlled so an evidence chip can open the trace. */
  open: boolean;
  onToggle: () => void;
  selectedEvidenceId: string | null;
  onSelectEvidence: (id: string | null) => void;
}

/** Qwen3 is a model name and stays verbatim; the other three are prose and are translated. */
const PLANNER_LABEL_KEYS: Record<string, TranslationKey> = {
  registry_rules: 'trace.registryFallback',
  template: 'trace.plannerTemplate',
  fixture: 'trace.plannerFixture',
};
const PLANNER_VERBATIM: Record<string, string> = { qwen3: 'Qwen3' };

function plannerLabel(t: Translate, value: string): string {
  if (PLANNER_VERBATIM[value]) return PLANNER_VERBATIM[value];
  const key = PLANNER_LABEL_KEYS[value];
  return key ? t(key) : value;
}

/** The backend sends a bare language code; name it in its own script when we know it. */
function languageName(code: string): string {
  return isLocaleCode(code) ? LOCALE_ENDONYM[code] : code;
}

function time(ts: string | undefined): string {
  if (!ts) return '';
  const d = new Date(ts.endsWith('Z') || ts.includes('+') ? ts : `${ts}Z`);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleTimeString();
}

function text(value: unknown): string {
  if (value == null) return '';
  return typeof value === 'string' ? value : JSON.stringify(value);
}

/** The region *type* comes from the backend and stays verbatim; only the wording is translated. */
function regionLabel(t: Translate, ev: EvidenceObject): string {
  if (!ev.spatial_region) return t('trace.noSpatialRegion');
  const suffix = ev.processing_parameters?.spatial_region_is_full_image
    ? t('trace.wholeImageFootprint')
    : t('trace.localizedRegion');
  return `${ev.spatial_region.type} · ${suffix}`;
}

/** Renders the stored execution trace of one job exactly as the backend recorded it. */
export default function ExecutionTrace({ trace, response, open, onToggle, selectedEvidenceId, onSelectEvidence }: ExecutionTraceProps) {
  const t = useT();

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
        <span className="exec-trace__toggle-label">{open ? t('trace.hide') : t('trace.show')}</span>
      </button>

      {open && (
        <div className="exec-trace__body rail rail--trace">
          {/* A query that changed language is shown before anything else: everything below ran on
              the English string, and a reviewer has to be able to see both. */}
          {response?.input_translation && (
            <section className="exec-trace__section rail__step">
              <h4><span className="rail__index">00</span>{t('trace.inputTranslation')}</h4>
              <p className="exec-trace__detail">
                {t('trace.translatedFrom', { lang: languageName(response.input_translation.from) })}
              </p>
              <dl className="exec-trace__kv">
                <dt>{t('trace.originalQuery')}</dt><dd>{response.input_translation.original}</dd>
                <dt>{t('trace.englishQuery')}</dt><dd>{response.input_translation.translated}</dd>
                <dt>engine</dt><dd>{response.input_translation.engine}</dd>
              </dl>
            </section>
          )}

          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">01</span>{t('trace.plannedTask')}</h4>
            {taskSpec ? (
              <dl className="exec-trace__kv">
                <dt>primary_task</dt><dd>{text(taskSpec.primary_task)}</dd>
                <dt>required_modalities</dt><dd>{text(taskSpec.required_modalities)}</dd>
                <dt>temporal_requirement</dt><dd>{text(taskSpec.temporal_requirement)}</dd>
              </dl>
            ) : (
              <p className="exec-trace__empty">{t('trace.noTask')}</p>
            )}
            {trace.planner && (
              <dl className="exec-trace__kv">
                {Object.entries(trace.planner).map(([k, v]) => (
                  <Fragment key={k}><dt>{k}</dt><dd>{v ? plannerLabel(t, v) : '—'}</dd></Fragment>
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
            <h4><span className="rail__index">02</span>{t('trace.executedTools')}</h4>
            {toolEvents.length === 0 ? (
              <p className="exec-trace__empty">{t('trace.noTools')}</p>
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
              <p className="exec-trace__detail">{t('trace.recoveries', { count: trace.replan_history.length })}</p>
            )}
          </section>

          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">03</span>{t('trace.verification')}</h4>
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
              <p className="exec-trace__empty">{t('trace.noVerification')}</p>
            )}
          </section>

          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">04</span>{t('trace.evidenceRegions')}</h4>
            {evidence.length === 0 ? (
              <p className="exec-trace__empty">{t('trace.noEvidence')}</p>
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
                        title={ev.spatial_region ? t('trace.highlightOnMap') : t('trace.noRegionForEvidence')}
                      >
                        <span className="evidence-card__claim">{rejectedIds.has(ev.evidence_id) && t('trace.rejectedPrefix')}{ev.claim}</span>
                        <span className="evidence-card__meta">
                          <span>{ev.evidence_id}</span>
                          <span>{ev.source_model}</span>
                          <span title={text(ev.processing_parameters?.confidence_source)}>{t('trace.modelConfidence')}: {(ev.confidence * 100).toFixed(0)}%</span>
                          <span>{regionLabel(t, ev)}</span>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          <section className="exec-trace__section rail__step">
            <h4><span className="rail__index">05</span>{t('trace.pipelineStages')}</h4>
            <ul className="exec-trace__list">
              {trace.trace.map((entry, i) => (
                <li key={i}>
                  <span className="exec-trace__time">{time(entry.timestamp)}</span> <strong>{entry.stage}</strong> — {entry.message}
                  {entry.error && <div className="exec-trace__detail">{entry.error}</div>}
                </li>
              ))}
            </ul>
          </section>

          <p className="exec-trace__detail exec-trace__english-note">{t('trace.englishNote')}</p>
        </div>
      )}
    </div>
  );
}
