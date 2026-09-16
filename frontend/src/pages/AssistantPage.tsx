import { useState, useCallback, useEffect, useRef } from 'react';
import UploadZone from '../components/UploadZone';
import QueryBox from '../components/QueryBox';
import AnalyzeButton from '../components/AnalyzeButton';
import TracePanel, { type TraceEntry } from '../components/results/TracePanel';
import ResultPanel from '../components/results/ResultPanel';
import type { UploadedFile } from '../components/UploadZone';
import ChatSidebar from '../components/assistant/ChatSidebar';
import ChatThread from '../components/assistant/ChatThread';
import { useChatSessions } from '../hooks/useChatSessions';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

const TERMINAL_STATES = [
  'DONE', 'FAILED', 'ABSTAIN', 'INSUFFICIENT_EVIDENCE', 'PRECONDITION_FAILED',
  'MODEL_UNAVAILABLE', 'INSUFFICIENT_OBSERVATIONS',
];

const EXAMPLE_QUERIES = [
  'What changed between these two images?',
  'Has a new airstrip been cleared between these two acquisitions?',
  'Is there a runway visible in this image?',
  'Describe this image.',
];

interface SystemStatus {
  llm: { backend: string; model: string; reachable: boolean | null };
  tools: { tool_id: string; name: string; available: boolean; reason: string }[];
}

export default function AssistantPage() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [query, setQuery] = useState('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Job states
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [trace, setTrace] = useState<TraceEntry[]>([]);
  const [result, setResult] = useState<any>(null);
  const [evidenceGraph, setEvidenceGraph] = useState<any>(null);
  const [system, setSystem] = useState<SystemStatus | null>(null);

  const chats = useChatSessions();
  // Session that submitted the in-flight query; its assistant turn is recorded when the job ends
  const pendingSessionRef = useRef<string | null>(null);

  // Re-open a finished or running job from a shareable link: ?job=<job_id>
  useEffect(() => {
    const sharedJob = new URLSearchParams(window.location.search).get('job');
    if (sharedJob) {
      setJobId(sharedJob);
      setLoading(true);
    }
  }, []);

  useEffect(() => {
    if (!jobId) return;
    const url = new URL(window.location.href);
    url.searchParams.set('job', jobId);
    window.history.replaceState(null, '', url.toString());
  }, [jobId]);

  useEffect(() => {
    fetch(`${API_BASE}/api/system`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setSystem)
      .catch(() => setSystem(null));
  }, []);

  const canAnalyze = files.length > 0 && query.trim().length > 0;

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    const pollJob = async () => {
      if (!jobId) return;

      try {
        const statusRes = await fetch(`${API_BASE}/api/jobs/${jobId}/status`);
        if (!statusRes.ok) return;
        const statusData = await statusRes.json();
        setStatus(statusData.status);

        const traceRes = await fetch(`${API_BASE}/api/jobs/${jobId}/trace`);
        if (traceRes.ok) {
          const traceData = await traceRes.json();
          setTrace(traceData.trace || []);
        }

        if (TERMINAL_STATES.includes(statusData.status)) {
          clearInterval(interval);

          const res = await fetch(`${API_BASE}/api/jobs/${jobId}/result`);
          const jobResult = res.ok ? (await res.json()).result : null;
          setResult(jobResult);

          if (pendingSessionRef.current) {
            chats.appendMessage(pendingSessionRef.current, {
              role: 'assistant',
              content: jobResult?.final_answer ?? `The request ended with status ${statusData.status}.`,
              confidence: typeof jobResult?.confidence === 'number'
                ? { value: jobResult.confidence, source: 'model_confidence_uncalibrated' }
                : undefined,
              status: statusData.status,
              jobId,
              timestamp: new Date().toISOString(),
            });
            pendingSessionRef.current = null;
          }

          const egRes = await fetch(`${API_BASE}/api/jobs/${jobId}/evidence_graph`);
          if (egRes.ok) setEvidenceGraph((await egRes.json()).evidence_graph);

          setLoading(false);
        }
      } catch (e) {
        console.error('Polling error', e);
      }
    };

    if (jobId && loading) {
      interval = setInterval(pollJob, 1500);
      pollJob();
    }

    return () => clearInterval(interval);
  }, [jobId, loading]);

  const handleAnalyze = useCallback(async () => {
    let hasError = false;

    if (files.length === 0) {
      setUploadError('Upload at least one image to analyze.');
      hasError = true;
    } else {
      setUploadError(null);
    }

    if (query.trim().length === 0) {
      setQueryError('Enter a query describing what you\'d like to analyze.');
      hasError = true;
    } else {
      setQueryError(null);
    }

    if (hasError) return;

    const sessionId = chats.ensureActiveSession();
    chats.appendMessage(sessionId, {
      role: 'user',
      content: query.trim(),
      attachments: files.map((f) => ({ name: f.file.name, size: f.file.size, type: f.file.type })),
      timestamp: new Date().toISOString(),
    });
    pendingSessionRef.current = sessionId;

    setLoading(true);
    setJobId(null);
    setStatus(null);
    setTrace([]);
    setResult(null);
    setEvidenceGraph(null);

    try {
      const formData = new FormData();
      files.forEach((f) => formData.append('files', f.file));
      formData.append('query', query.trim());

      const response = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setJobId(data.job_id);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setUploadError(
        message === 'Failed to fetch'
          ? `Cannot reach the backend at ${API_BASE}. Check that uvicorn is running and that it allows this page's origin (${window.location.origin}).`
          : message
      );
      chats.appendMessage(sessionId, {
        role: 'assistant',
        content: 'The request could not be sent to the backend, so no analysis was run.',
        status: 'FAILED',
        timestamp: new Date().toISOString(),
      });
      pendingSessionRef.current = null;
      setLoading(false);
    }
  }, [files, query, chats]);

  /** Clears the live job view (used when switching chats) without touching stored history. */
  const resetWorkspace = useCallback(() => {
    setJobId(null);
    setStatus(null);
    setTrace([]);
    setResult(null);
    setEvidenceGraph(null);
    setUploadError(null);
    setQueryError(null);
    const url = new URL(window.location.href);
    if (url.searchParams.has('job')) {
      url.searchParams.delete('job');
      window.history.replaceState(null, '', url.toString());
    }
  }, []);

  const handleNewChat = useCallback(() => {
    if (loading) return;
    chats.newChat();
    resetWorkspace();
    setFiles([]);
    setQuery('');
  }, [loading, chats, resetWorkspace]);

  const handleSelectSession = useCallback((id: string) => {
    if (loading || id === chats.activeId) return;
    chats.selectSession(id);
    resetWorkspace();
  }, [loading, chats, resetWorkspace]);

  const isTerminal = status !== null && TERMINAL_STATES.includes(status);

  return (
    <div className={`assistant-layout ${chats.collapsed ? 'assistant-layout--collapsed' : ''}`}>
      <ChatSidebar
        sessions={chats.sessions}
        activeId={chats.activeId}
        collapsed={chats.collapsed}
        busy={loading}
        onToggleCollapsed={() => chats.setCollapsed(!chats.collapsed)}
        onNewChat={handleNewChat}
        onSelect={handleSelectSession}
        onDelete={chats.deleteSession}
      />

      <div className="assistant-center">
      <ChatThread session={chats.activeSession} />

      <main className="app-main">
        {/* ── Left Panel: Inputs ── */}
        <section className="panel panel--inputs">
          <div className="panel__title">
            Inputs
            {system && (
              <span className={`pill panel__title-pill ${system.llm.reachable === false ? 'pill--warn' : ''}`} title={system.llm.backend}>
                LLM: {system.llm.model}{system.llm.reachable === false ? ' · offline (registry rules)' : ''}
              </span>
            )}
          </div>

          <UploadZone
            files={files}
            onFilesChange={setFiles}
            error={uploadError}
            onError={setUploadError}
          />

          <QueryBox
            query={query}
            onChange={(v) => {
              setQuery(v);
              if (v.trim().length > 0) setQueryError(null);
            }}
            error={queryError}
          />

          <div className="example-queries">
            {EXAMPLE_QUERIES.map((q) => (
              <button key={q} className="example-query" onClick={() => { setQuery(q); setQueryError(null); }}>
                {q}
              </button>
            ))}
          </div>

          <AnalyzeButton
            disabled={!canAnalyze}
            loading={loading}
            onClick={handleAnalyze}
          />

          {system && (
            <div className="engine-list">
              <div className="engine-list__title">Specialist engines (Tool Registry)</div>
              {system.tools.map((t) => (
                <div key={t.tool_id} className="engine-list__item" title={t.reason}>
                  <span className={`dot ${t.available ? 'dot--ok' : 'dot--off'}`} />
                  <span>{t.name}</span>
                  {!t.available && <span className="engine-list__reason">{t.reason.split(':')[0]}</span>}
                </div>
              ))}
            </div>
          )}
        </section>

        {/* ── Right Panel: Results & Trace ── */}
        <section className="panel panel--results">
          <div className="panel__title">Analysis & Reporting</div>
          {jobId && (
            <div className="job-status-container">
              {isTerminal ? (
                <ResultPanel
                  apiBase={API_BASE}
                  status={status as string}
                  result={result}
                  evidenceGraph={evidenceGraph}
                  jobId={jobId}
                />
              ) : (
                <div className="result-panel result-panel--processing">
                  <div className="spinner"></div>
                  <p>Processing ({status || 'QUEUED'})…</p>
                </div>
              )}
              <div style={{ marginTop: '20px' }}>
                <TracePanel trace={trace} />
              </div>
            </div>
          )}

          {!jobId && !loading && (
            <div className="empty-state">
              <p>Upload images and enter a query to begin analysis.</p>
              <p className="empty-state__hint">For change detection, upload the earlier image first (Image 1 = before, Image 2 = after) unless the files carry acquisition dates.</p>
            </div>
          )}
        </section>
      </main>
      </div>
    </div>
  );
}
