import { useState, useCallback, useEffect } from 'react';
import UploadZone from '../components/UploadZone';
import QueryBox from '../components/QueryBox';
import AnalyzeButton from '../components/AnalyzeButton';
import TracePanel, { type TraceEntry } from '../components/results/TracePanel';
import ResultPanel from '../components/results/ResultPanel';
import type { UploadedFile } from '../components/UploadZone';

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
          if (res.ok) setResult((await res.json()).result);

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
      setLoading(false);
    }
  }, [files, query]);

  const isTerminal = status !== null && TERMINAL_STATES.includes(status);

  return (
    <>

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
    </>
  );
}
