import { useState, useCallback, useEffect } from 'react';
import UploadZone from './components/UploadZone';
import QueryBox from './components/QueryBox';
import AnalyzeButton from './components/AnalyzeButton';
import TracePanel, { type TraceEntry } from './components/results/TracePanel';
import ResultPanel from './components/results/ResultPanel';
import type { UploadedFile } from './components/UploadZone';

const API_BASE = 'http://localhost:8000';

export default function App() {
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

  const canAnalyze = files.length > 0 && query.trim().length > 0;

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    const pollJob = async () => {
      if (!jobId) return;
      
      try {
        // Poll status
        const statusRes = await fetch(`${API_BASE}/api/jobs/${jobId}/status`);
        if (statusRes.ok) {
          const statusData = await statusRes.json();
          setStatus(statusData.status);
          
          // Poll trace
          const traceRes = await fetch(`${API_BASE}/api/jobs/${jobId}/trace`);
          if (traceRes.ok) {
            const traceData = await traceRes.json();
            setTrace(traceData.trace || []);
          }
          
          const terminalStates = ["DONE", "FAILED", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "PRECONDITION_FAILED"];
          if (terminalStates.includes(statusData.status)) {
            clearInterval(interval);
            setLoading(false);
            
            // Fetch results if any
            if (statusData.status === "DONE") {
              const res = await fetch(`${API_BASE}/api/jobs/${jobId}/result`);
              if (res.ok) setResult((await res.json()).result);
              
              const egRes = await fetch(`${API_BASE}/api/jobs/${jobId}/evidence_graph`);
              if (egRes.ok) setEvidenceGraph((await egRes.json()).evidence_graph);
            }
          }
        }
      } catch (e) {
        console.error("Polling error", e);
      }
    };
    
    if (jobId && loading) {
      interval = setInterval(pollJob, 1000);
      pollJob(); // initial call
    }
    
    return () => clearInterval(interval);
  }, [jobId, loading]);

  const handleAnalyze = useCallback(async () => {
    // Client-side validation
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
      setUploadError(
        err instanceof Error ? err.message : 'Failed to connect to backend.'
      );
      setLoading(false);
    }
  }, [files, query]);

  return (
    <>
      <header className="app-header">
        <div className="app-header__logo">
          <div className="app-header__icon">🛰</div>
          <div>
            <div className="app-header__title">SatQuery AI</div>
            <div className="app-header__subtitle">
              Agentic Remote-Sensing Assistant
            </div>
          </div>
        </div>
        <span className="app-header__version">v0.1.0</span>
      </header>

      <main className="app-main">
        {/* ── Left Panel: Inputs ── */}
        <section className="panel panel--inputs">
          <div className="panel__title">Inputs</div>

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

          <AnalyzeButton
            disabled={!canAnalyze}
            loading={loading}
            onClick={handleAnalyze}
          />
        </section>

        {/* ── Right Panel: Results & Trace ── */}
        <section className="panel panel--results">
          <div className="panel__title">Analysis & Reporting</div>
          {jobId && (
            <div className="job-status-container">
              <div style={{ marginBottom: '20px' }}>
                <TracePanel trace={trace} />
              </div>
              
              {(status && status !== 'QUEUED' && status !== 'MC1_VALIDATING' && status !== 'MC2_PARSING' && status !== 'MC3_PLANNING' && status !== 'MC4_EXECUTING' && status !== 'MC5_NORMALIZING' && status !== 'MC6_VERIFYING' && status !== 'MC6_REPLANNING' && status !== 'MC7_ANSWERING') || status === 'DONE' ? (
                <ResultPanel 
                  status={status} 
                  result={result} 
                  evidenceGraph={evidenceGraph} 
                  files={files} 
                />
              ) : null}
            </div>
          )}
          
          {!jobId && !loading && (
            <div className="empty-state">
              <p>Upload images and enter a query to begin analysis.</p>
            </div>
          )}
        </section>
      </main>
    </>
  );
}
