import { useI18n } from '../i18n/useT';
import { useState, useCallback, useEffect } from 'react';
import type { UploadedFile } from '../components/UploadZone';
import ChatSidebar from '../components/assistant/ChatSidebar';
import ChatThread from '../components/assistant/ChatThread';
import ChatComposer from '../components/assistant/ChatComposer';
import ModelSelector from '../components/assistant/ModelSelector';
import { useChatSessions } from '../hooks/useChatSessions';
import type { ChatMessage } from '../lib/chatStorage';
import { snapshotJob, TERMINAL_STATES } from '../lib/jobResponse';
import { loadSampleFiles, type SampleQuery } from '../lib/samples';

// Backend base URL from the build-time env var (see frontend/.env.example); trailing slashes are tolerated
const API_BASE = (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/+$/, '');
const POLL_MS = 1500;
const MAX_POLL_FAILURES = 20; // ~30 s without contact


const STAGE_LABELS: Record<string, string> = {
  QUEUED: 'queued',
  MC1_VALIDATING: 'checking the inputs',
  QUERY_INTELLIGENCE: 'understanding the question',
  TOOL_SELECTION: 'selecting specialist models',
  OBSERVATION_BINDING: 'binding the images to the task',
  WORKFLOW_PLANNING: 'planning the workflow',
  AGENTIC_EXECUTION: 'running specialist models',
  EVIDENCE_NORMALIZATION: 'collecting evidence',
  VERIFICATION: 'verifying evidence',
  ANSWER_SYNTHESIS: 'writing the answer',
};

interface SystemStatus {
  llm: { backend: string; model: string; reachable: boolean | null };
  tools: { tool_id: string; name: string; available: boolean; reason: string }[];
}

interface PendingJob {
  jobId: string;
  sessionId: string;
}


export default function AssistantPage() {
  const { t, locale } = useI18n();
  const chats = useChatSessions();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [prompt, setPrompt] = useState('');
  const [composerError, setComposerError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [pending, setPending] = useState<PendingJob | null>(null);
  const [stage, setStage] = useState<string>('QUEUED');
  const [system, setSystem] = useState<SystemStatus | null>(null);
  // Narrow screens show the history as an off-canvas drawer; it starts closed and is not persisted.
  const [drawerOpen, setDrawerOpen] = useState(false);

  const inFlight = submitting || pending !== null;
  const { appendMessage } = chats;

  useEffect(() => {
    fetch(`${API_BASE}/api/system`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setSystem)
      .catch(() => setSystem(null));
  }, []);

  // Re-open a job from a shareable link (?job=<id>) as an assistant turn in a fresh chat
  useEffect(() => {
    const sharedJob = new URLSearchParams(window.location.search).get('job');
    if (sharedJob) {
      const sessionId = chats.newChat();
      setPending({ jobId: sharedJob, sessionId });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (pending) url.searchParams.set('job', pending.jobId);
    else url.searchParams.delete('job');
    window.history.replaceState(null, '', url.toString());
  }, [pending]);

  const finishWith = useCallback((job: PendingJob, message: ChatMessage) => {
    appendMessage(job.sessionId, message);
    setPending(null);
    setStage('QUEUED');
  }, [appendMessage]);

  // Poll the in-flight job; one completed payload is rendered when it reaches a terminal status
  useEffect(() => {
    if (!pending) return;
    let cancelled = false;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const statusRes = await fetch(`${API_BASE}/api/jobs/${pending.jobId}/status`);
        if (statusRes.status === 404) {
          if (!cancelled) {
            finishWith(pending, {
              role: 'assistant',
              content: t('error.jobGone'),
              status: 'FAILED',
              error: 'job_not_found',
              jobId: pending.jobId,
              timestamp: new Date().toISOString(),
            });
          }
          return;
        }
        if (!statusRes.ok) throw new Error(`status ${statusRes.status}`);
        const { status } = await statusRes.json();
        failures = 0;
        if (cancelled) return;
        setStage(status);

        if (TERMINAL_STATES.includes(status)) {
          const [resultRes, traceRes, structuredRes] = await Promise.all([
            fetch(`${API_BASE}/api/jobs/${pending.jobId}/result`),
            fetch(`${API_BASE}/api/jobs/${pending.jobId}/trace`),
            fetch(`${API_BASE}/api/jobs/${pending.jobId}/structured_trace`),
          ]);
          const result = resultRes.ok ? (await resultRes.json()).result : null;
          const progress = traceRes.ok ? (await traceRes.json()).trace : [];
          const structured = structuredRes.ok ? (await structuredRes.json()).structured_trace : null;
          if (cancelled) return;

          const { response, executionTrace } = snapshotJob(result, progress, structured);
          const content = status === 'FAILED'
            ? t('error.serverFailed')
            : response.final_answer || t('error.endedWithStatus', { status });
          finishWith(pending, {
            role: 'assistant',
            content,
            status,
            jobId: pending.jobId,
            confidence: status === 'DONE' ? { value: response.confidence, source: 'model_confidence_uncalibrated' } : undefined,
            response,
            executionTrace,
            timestamp: new Date().toISOString(),
          });
          return;
        }
      } catch {
        failures += 1;
        if (failures >= MAX_POLL_FAILURES) {
          if (!cancelled) {
            finishWith(pending, {
              role: 'assistant',
              content: t('error.lostContact'),
              status: 'FAILED',
              error: 'backend_unreachable',
              jobId: pending.jobId,
              timestamp: new Date().toISOString(),
            });
          }
          return;
        }
      }
      if (!cancelled) timer = setTimeout(poll, POLL_MS);
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [pending, finishWith]);

  const submit = useCallback(async (sendFiles: UploadedFile[], question: string) => {
    if (inFlight) return;
    if (sendFiles.length === 0) {
      setComposerError(t('error.noImage'));
      return;
    }
    if (!question) {
      setComposerError(t('error.noQuestion'));
      return;
    }
    setComposerError(null);

    const sessionId = chats.ensureActiveSession();
    appendMessage(sessionId, {
      role: 'user',
      content: question,
      attachments: sendFiles.map((f) => ({ name: f.file.name, size: f.file.size, type: f.file.type })),
      timestamp: new Date().toISOString(),
    });

    setSubmitting(true);
    try {
      const formData = new FormData();
      sendFiles.forEach((f) => formData.append('files', f.file));
      formData.append('query', question);
      // The language `question` is written in. The backend translates it to English before MC2
      // and records the transformation in the trace; "en" is the default and changes nothing.
      formData.append('query_language', locale);
      let response: Response;
      try {
        response = await fetch(`${API_BASE}/api/query`, { method: 'POST', body: formData });
      } catch {
        throw new Error('unreachable');
      }
      if (!response.ok) {
        // The server answered but refused the request (e.g. 413 upload too large): show its reason, not "unreachable"
        const detail = await response.json().then((b) => (typeof b?.detail === 'string' ? b.detail : null)).catch(() => null);
        appendMessage(sessionId, {
          role: 'assistant',
          content: detail ?? `The backend rejected the request (HTTP ${response.status}), so no analysis was run.`,
          status: 'FAILED',
          error: `http_${response.status}`,
          timestamp: new Date().toISOString(),
        });
        return;
      }
      const data = await response.json();
      setStage('QUEUED');
      setPending({ jobId: data.job_id, sessionId });
      sendFiles.forEach((f) => f.previewUrl && URL.revokeObjectURL(f.previewUrl));
      setFiles([]);
      setPrompt('');
    } catch {
      appendMessage(sessionId, {
        role: 'assistant',
        content: t('error.unreachable'),
        status: 'FAILED',
        error: 'backend_unreachable',
        timestamp: new Date().toISOString(),
      });
    } finally {
      setSubmitting(false);
    }
  }, [inFlight, chats, appendMessage]);

  const handleSend = useCallback(() => submit(files, prompt.trim()), [submit, files, prompt]);

  /** Sample query: load the imagery bundled with the site and run the real pipeline on it. */
  const handleRunSample = useCallback(async (sample: SampleQuery) => {
    if (inFlight) return;
    setComposerError(null);
    try {
      const loaded = await loadSampleFiles(sample);
      const attachments: UploadedFile[] = loaded.map((file, i) => ({
        file,
        id: `${sample.id}-${i}`,
        previewUrl: null,
        isGeoTiff: file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff'),
      }));
      await submit(attachments, sample.query);
    } catch {
      setComposerError(t('error.sampleLoad'));
    }
  }, [inFlight, submit]);

  const handleNewChat = useCallback(() => {
    if (inFlight) return;
    chats.newChat();
    setComposerError(null);
    setDrawerOpen(false);
  }, [inFlight, chats]);

  const handleSelectSession = useCallback((id: string) => {
    if (inFlight || id === chats.activeId) return;
    chats.selectSession(id);
    setComposerError(null);
    setDrawerOpen(false);
  }, [inFlight, chats]);

  const availableTools = system?.tools.filter((t) => t.available) ?? [];
  const showingProgress = pending !== null && pending.sessionId === chats.activeId;
  const pendingLabel = showingProgress
    ? (STAGE_LABELS[stage] ?? stage.toLowerCase())
    : submitting ? 'sending the request (a sleeping server can take about a minute to wake up)' : null;
  // The raw backend stage drives the live pipeline view; null while only the POST is in flight
  const pendingStage = showingProgress ? stage : null;

  return (
    <div className={`assistant-layout ${chats.collapsed ? 'assistant-layout--collapsed' : ''}`}>
      <ChatSidebar
        sessions={chats.sessions}
        activeId={chats.activeId}
        collapsed={chats.collapsed}
        drawerOpen={drawerOpen}
        busy={inFlight}
        onToggleCollapsed={() => chats.setCollapsed(!chats.collapsed)}
        onCloseDrawer={() => setDrawerOpen(false)}
        onNewChat={handleNewChat}
        onSelect={handleSelectSession}
        onDelete={chats.deleteSession}
      />

      {drawerOpen && (
        <button className="sidebar-scrim" aria-label={t('sidebar.closeDrawer')} onClick={() => setDrawerOpen(false)} />
      )}

      <div className="assistant-center">
        <div className="chat-header">
          <button
            className="chat-header__menu"
            onClick={() => setDrawerOpen(true)}
            aria-label={t('sidebar.openDrawer')}
            title={t('sidebar.aria')}
          >
            ☰
          </button>
          <ModelSelector />
          {system && (
            <span
              className={`pill ${system.llm.reachable === false ? 'pill--warn' : ''}`}
              title={system.tools.map((t) => `${t.name}: ${t.available ? 'available' : t.reason}`).join('\n')}
            >
              {system.llm.reachable === false ? 'LLM offline · Tool Registry rules' : `LLM: ${system.llm.model}`}
              {' · '}{availableTools.length} of {system.tools.length} specialist engines available
            </span>
          )}
        </div>

        <ChatThread
          apiBase={API_BASE}
          session={chats.activeSession}
          pendingStage={pendingStage}
          pendingLabel={pendingLabel}
          tools={system?.tools ?? null}
          busy={inFlight}
          onRunSample={handleRunSample}
        />

        <ChatComposer
          files={files}
          onFilesChange={setFiles}
          prompt={prompt}
          onPromptChange={(v) => { setPrompt(v); if (v.trim()) setComposerError(null); }}
          onSend={handleSend}
          inFlight={inFlight}
          error={composerError}
          onError={setComposerError}
        />
      </div>
    </div>
  );
}
