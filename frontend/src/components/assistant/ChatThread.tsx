import { useEffect, useRef, useState } from 'react';
import type { ChatMessage, ChatSession } from '../../lib/chatStorage';
import { hasSpatialEvidence } from '../../lib/jobResponse';
import BeforeAfterViewer from '../results/BeforeAfterViewer';
import ExecutionTrace from './ExecutionTrace';

interface ChatThreadProps {
  apiBase: string;
  session: ChatSession | null;
  /** Stage label of the in-flight job for this session, if any. */
  pendingStage: string | null;
  examples: string[];
  onPickExample: (query: string) => void;
}

export function attachmentLabel(index: number, count: number): string {
  return count === 2 ? `Image ${index === 0 ? 'A' : 'B'}` : 'Image';
}

const STOP_TITLES: Record<string, string> = {
  PRECONDITION_FAILED: 'Inputs rejected',
  INSUFFICIENT_OBSERVATIONS: 'Inputs do not fit the request',
  ABSTAIN: 'No answer given',
  INSUFFICIENT_EVIDENCE: 'Insufficient evidence',
  MODEL_UNAVAILABLE: 'Model unavailable',
  FAILED: 'Analysis failed',
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
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const response = message.response;
  const status = message.status ?? 'DONE';
  const spatial = hasSpatialEvidence(response);
  const firstPreview = spatial && response?.observations ? Object.values(response.observations)[0]?.preview_url : null;
  const imageryAvailable = useImageAvailable(firstPreview ? `${apiBase}${firstPreview}` : null);
  // Reasons are listed only when the answer text does not already spell them out
  const reasons = (response?.failed ?? []).map((f) => f.reason).filter((r) => r && !message.content.includes(r));
  const showConfidence = status === 'DONE' && !!message.confidence && (response?.evidence_objects.length ?? 0) > 0;

  return (
    <div className="chat-turn chat-turn--assistant">
      <div className={`chat-turn__bubble ${status !== 'DONE' ? 'chat-turn__bubble--stopped' : ''}`}>
        {status !== 'DONE' && <div className="chat-turn__status">{STOP_TITLES[status] ?? status} · {status}</div>}

        {/* 1. Answer */}
        <div className="chat-turn__content">{message.content}</div>

        {reasons.length > 0 && (
          <ul className="chat-turn__reasons">{reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        )}

        {/* 2. Confidence — raw model confidence; MC6.2 calibration is not implemented */}
        {showConfidence && message.confidence && (
          <div className="chat-turn__confidence" title={response?.confidence_note}>
            model confidence (uncalibrated): {(message.confidence.value * 100).toFixed(0)}%
          </div>
        )}

        {/* 3. Caveats */}
        {response && response.caveats.length > 0 && (
          <div className="caveats-section chat-turn__caveats">
            <strong>Caveats</strong>
            <ul>{response.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
          </div>
        )}

        {/* 4. Spatial evidence — omitted entirely when the backend produced none */}
        {spatial && response && imageryAvailable === true && (
          <div className="chat-turn__spatial">
            <BeforeAfterViewer
              apiBase={apiBase}
              result={response}
              evidenceGraph={{ nodes: response.evidence_objects.map((ev) => ({ id: ev.evidence_id, type: 'evidence', data: ev })) }}
              selectedEvidenceId={selectedEvidenceId}
            />
          </div>
        )}
        {spatial && imageryAvailable === false && (
          <div className="chat-turn__note">
            The imagery for this answer is no longer held by the backend (jobs are kept in server memory), so the map cannot be redrawn.
          </div>
        )}

        {message.executionTrace && (
          <ExecutionTrace
            trace={message.executionTrace}
            response={response}
            selectedEvidenceId={selectedEvidenceId}
            onSelectEvidence={setSelectedEvidenceId}
          />
        )}

        {message.jobId && status === 'DONE' && (
          <div className="chat-turn__exports">
            {[['pdf', 'PDF report'], ['geojson', 'GeoJSON'], ['json', 'JSON trace']].map(([fmt, label]) => (
              <a key={fmt} href={`${apiBase}/api/jobs/${message.jobId}/export/${fmt}`} className="export-link">{label}</a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatThread({ apiBase, session, pendingStage, examples, onPickExample }: ChatThreadProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const messageCount = session?.messages.length ?? 0;

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [messageCount, pendingStage, session?.id]);

  if (!session || (messageCount === 0 && !pendingStage)) {
    return (
      <section className="chat-thread chat-thread--empty" aria-label="New conversation">
        <div className="chat-empty">
          <h2 className="chat-empty__title">Ask about your imagery</h2>
          <p className="chat-empty__hint">
            Attach one or two images (GeoTIFF, PNG or JPEG) and ask a question. For change detection, attach the
            earlier image first — Image A is treated as before and Image B as after unless the files carry acquisition dates.
          </p>
          <div className="example-queries">
            {examples.map((q) => (
              <button key={q} className="example-query" onClick={() => onPickExample(q)}>{q}</button>
            ))}
          </div>
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
                      <span className="chat-chip__label">{attachmentLabel(i, message.attachments!.length)}</span>
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

      {pendingStage && (
        <div className="chat-turn chat-turn--assistant" aria-live="polite">
          <div className="chat-turn__bubble chat-turn__bubble--pending">
            <span className="chat-spinner" aria-hidden="true" />
            <span>Analyzing… {pendingStage}</span>
          </div>
        </div>
      )}
      <div ref={endRef} />
    </section>
  );
}
