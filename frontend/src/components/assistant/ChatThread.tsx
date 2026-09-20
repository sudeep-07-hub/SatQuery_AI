import { useEffect, useRef, useState } from 'react';
import type { ChatMessage, ChatSession } from '../../lib/chatStorage';
import { hasSpatialEvidence } from '../../lib/jobResponse';
import { stepIndexForStage } from '../../lib/pipeline';
import type { SampleQuery } from '../../lib/samples';
import BeforeAfterViewer from '../results/BeforeAfterViewer';
import PipelineRail from '../PipelineRail';
import ExecutionTrace from './ExecutionTrace';
import ConfidenceBadge from './ConfidenceBadge';
import EvidenceChips from './EvidenceChips';
import SampleQueries from './SampleQueries';
import { useT } from '../../i18n/useT';
import type { Translate } from '../../i18n/I18nProvider';
import type { TranslationKey } from '../../i18n/types';

interface ChatThreadProps {
  apiBase: string;
  session: ChatSession | null;
  /** Backend stage of the in-flight job for this session (progress_trace stage), if any. */
  pendingStage: string | null;
  pendingLabel: string | null;
  tools: { tool_id: string; available: boolean; reason: string }[] | null;
  busy: boolean;
  onRunSample: (sample: SampleQuery) => void;
}

export function attachmentLabel(t: Translate, index: number, count: number): string {
  if (count !== 2) return t('thread.image');
  return index === 0 ? t('thread.imageA') : t('thread.imageB');
}

/**
 * Human gloss for each stop status. The enum is the auditable value and is printed verbatim next
 * to the gloss, never replaced by it. An unrecognised status falls back to the enum alone.
 */
const STOP_TITLE_KEYS: Record<string, TranslationKey> = {
  PRECONDITION_FAILED: 'stop.PRECONDITION_FAILED',
  INSUFFICIENT_OBSERVATIONS: 'stop.INSUFFICIENT_OBSERVATIONS',
  ABSTAIN: 'stop.ABSTAIN',
  INSUFFICIENT_EVIDENCE: 'stop.INSUFFICIENT_EVIDENCE',
  MODEL_UNAVAILABLE: 'stop.MODEL_UNAVAILABLE',
  FAILED: 'stop.FAILED',
  TRANSLATION_UNAVAILABLE: 'stop.TRANSLATION_UNAVAILABLE',
};

/** Spatial evidence imagery is served from the backend's in-memory job store; check it is still there. */
function useImageAvailable(url: string | null): boolean | null {
  const [available, setAvailable] = useState<boolean | null>(null);
  useEffect(() => {
    if (!url) return;
    const img = new Image();
    img.onload = () => setAvailable(true);
    img.onerror = () => setAvailable(false);
    img.src = url;
  }, [url]);
  return url ? available : false;
}

function AssistantTurn({ apiBase, message }: { apiBase: string; message: ChatMessage }) {
  const t = useT();
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const response = message.response;
  const status = message.status ?? 'DONE';
  const spatial = hasSpatialEvidence(response);
  const firstPreview = spatial && response?.observations ? Object.values(response.observations)[0]?.preview_url : null;
  const imageryAvailable = useImageAvailable(firstPreview ? `${apiBase}${firstPreview}` : null);
  const reasons = (response?.failed ?? []).map((f) => f.reason).filter((r) => r && !message.content.includes(r));
  const evidence = response?.evidence_objects ?? [];
  // Rejected claims come from the stored verification_result (real backend field)
  const rejectedIds = new Set<string>((message.executionTrace?.verification_result?.rejected_claims ?? []).map((c) => c.evidence_id));
  const showConfidence = status === 'DONE' && !!message.confidence && evidence.length > 0;

  const selectEvidence = (id: string | null) => {
    setSelectedEvidenceId(id);
    if (id) setTraceOpen(true);
  };

  return (
    <div className="chat-turn chat-turn--assistant">
      <div className={`chat-turn__bubble ${status !== 'DONE' ? 'chat-turn__bubble--stopped' : ''}`}>
        {status !== 'DONE' && (
          <div className="chat-turn__status">
            {STOP_TITLE_KEYS[status] ? `${t(STOP_TITLE_KEYS[status])} · ${status}` : status}
          </div>
        )}

        {/* 1. Answer */}
        <div className="chat-turn__content">{message.content}</div>

        {reasons.length > 0 && (
          <ul className="chat-turn__reasons">{reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        )}

        {/* 2. Confidence — raw model confidence; MC6.2 calibration is not implemented */}
        {showConfidence && message.confidence && (
          <ConfidenceBadge confidence={message.confidence} note={response?.confidence_note} />
        )}

        {/* Evidence behind the answer (MC5 objects referenced by MC7.1) */}
        {status === 'DONE' && (
          <EvidenceChips
            evidence={evidence}
            rejectedIds={rejectedIds}
            selectedEvidenceId={selectedEvidenceId}
            onSelect={selectEvidence}
          />
        )}

        {/* 3. Caveats */}
        {response && response.caveats.length > 0 && (
          <div className="caveats-section chat-turn__caveats">
            <strong>{t('thread.caveats')}</strong>
            <ul>{response.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
          </div>
        )}

        {/* 4. Spatial evidence — omitted entirely when the backend produced none */}
        {spatial && response && imageryAvailable === true && (
          <div className="chat-turn__spatial">
            <BeforeAfterViewer
              apiBase={apiBase}
              result={response}
              evidenceGraph={{ nodes: evidence.map((ev) => ({ id: ev.evidence_id, type: 'evidence', data: ev })) }}
              selectedEvidenceId={selectedEvidenceId}
            />
          </div>
        )}
        {spatial && imageryAvailable === false && (
          <div className="chat-turn__note">{t('thread.imageryGone')}</div>
        )}

        {message.executionTrace && (
          <ExecutionTrace
            trace={message.executionTrace}
            response={response}
            open={traceOpen}
            onToggle={() => setTraceOpen((v) => !v)}
            selectedEvidenceId={selectedEvidenceId}
            onSelectEvidence={setSelectedEvidenceId}
          />
        )}

        {message.jobId && status === 'DONE' && (
          <div className="chat-turn__exports">
            {([['pdf', 'export.pdf'], ['geojson', 'export.geojson'], ['json', 'export.json']] as const).map(([fmt, key]) => (
              <a key={fmt} href={`${apiBase}/api/jobs/${message.jobId}/export/${fmt}`} className="export-link">{t(key)}</a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatThread({
  apiBase, session, pendingStage, pendingLabel, tools, busy, onRunSample,
}: ChatThreadProps) {
  const t = useT();
  const endRef = useRef<HTMLDivElement>(null);
  const messageCount = session?.messages.length ?? 0;

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [messageCount, pendingStage, session?.id]);

  if (!session || (messageCount === 0 && !pendingLabel)) {
    return (
      <section className="chat-thread chat-thread--empty" aria-label={t('thread.newConversation')}>
        <div className="chat-empty">
          <h2 className="chat-empty__title">Ask about your imagery</h2>
          <SampleQueries tools={tools} busy={busy} onRun={onRunSample} />
          <p className="chat-empty__hint">
            Or attach your own: one or two images (GeoTIFF, PNG or JPEG). For change detection, attach the
            earlier image first — Image A is treated as before and Image B as after unless the files carry
            acquisition dates.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="chat-thread" aria-label={`Conversation: ${session.title}`}>
      {session.messages.map((message, idx) =>
        message.role === 'user' ? (
          <div key={`${message.timestamp}-${idx}`} className="chat-turn chat-turn--user">
            <div className="chat-turn__bubble">
              <div className="chat-turn__content">{message.content}</div>
              {message.attachments && message.attachments.length > 0 && (
                <div className="chat-chips">
                  {message.attachments.map((a, i) => (
                    <span key={`${a.name}-${i}`} className="chat-chip">
                      <span className="chat-chip__label">{attachmentLabel(t, i, message.attachments!.length)}</span>
                      <span className="chat-chip__name">{a.name}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <AssistantTurn key={`${message.timestamp}-${idx}`} apiBase={apiBase} message={message} />
        )
      )}

      {pendingLabel && (
        <div className="chat-turn chat-turn--assistant" aria-live="polite">
          <div className="chat-turn__bubble chat-turn__bubble--progress">
            <div className="chat-progress__head">
              <span className="chat-spinner" aria-hidden="true" />
              <span>{pendingLabel}</span>
            </div>
            {/* Driven by the stage the backend reports; no timed animation */}
            <PipelineRail
              variant="progress"
              activeStep={stepIndexForStage(pendingStage)}
              activeStageLabel={pendingStage}
            />
          </div>
        </div>
      )}
      <div ref={endRef} />
    </section>
  );
}
