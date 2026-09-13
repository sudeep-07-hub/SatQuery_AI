import React from 'react';
import BeforeAfterViewer from './BeforeAfterViewer';

interface ResultPanelProps {
  status: string;
  result: any;
  evidenceGraph: any;
  files: any[];
  jobId: string;
}

export default function ResultPanel({ status, result, evidenceGraph, files, jobId }: ResultPanelProps) {
  if (["FAILED", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "PRECONDITION_FAILED"].includes(status)) {
    let title = "Analysis Terminated";
    let desc = "The agentic controller determined that this request could not be confidently completed.";
    
    if (status === "PRECONDITION_FAILED") {
      title = "Precondition Failed";
      desc = "The input images do not meet the minimum requirements for the selected tool. " + 
             "This could be due to missing CRS, insufficient overlap, or an unsupported modality combination (e.g., SAR + SAR for change detection).";
    } else if (status === "ABSTAIN") {
      title = "Model Abstained";
      desc = "No viable workflow plan could be formed. The system abstained rather than guessing.";
    } else if (status === "INSUFFICIENT_EVIDENCE") {
      title = "Insufficient Evidence";
      desc = "The verifier rejected the findings due to low confidence or conflicting evidence across multiple replan attempts.";
    }

    return (
      <div className={`result-panel result-panel--error result-panel--${status.toLowerCase()}`}>
        <h2>{title}</h2>
        <div className="error-status">Status: {status}</div>
        <p className="error-description">{desc}</p>
        
        {result?.failed && result.failed.length > 0 && (
          <div className="error-details" style={{ marginTop: '10px', padding: '10px', background: '#fef2f2', borderRadius: '4px', fontSize: '14px' }}>
            <strong>Specific reasons:</strong>
            <ul style={{ margin: '5px 0 0 20px', color: '#b91c1c' }}>
              {result.failed.map((f: any, i: number) => (
                <li key={i}>{f.reason || f.check}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  if (status === "DONE" && result) {
    const confidence = result.confidence || 0;
    let confLabel = 'Low';
    let confColor = '#ef4444'; // red
    if (confidence > 0.8) {
      confLabel = 'High';
      confColor = '#22c55e'; // green
    } else if (confidence > 0.5) {
      confLabel = 'Medium';
      confColor = '#eab308'; // yellow
    }

    const hasNonLocalized = result.caveats && result.caveats.includes("Single-source unverified evidence.");
    
    // Extract evidence objects from the graph
    const evidenceNodes = evidenceGraph?.nodes?.filter((n: any) => n.type === 'evidence') || [];
    const [selectedEvidenceId, setSelectedEvidenceId] = React.useState<string | null>(null);

    return (
      <div className="result-panel">
        <div className="result-header">
          <h2>Analysis Result</h2>
          <div style={{ display: 'flex', gap: '10px' }}>
            {hasNonLocalized && (
              <div className="badge" style={{ padding: '4px 8px', borderRadius: '12px', fontSize: '12px', fontWeight: 'bold', background: '#f3f4f6', color: '#4b5563', border: '1px solid #d1d5db' }}>
                Non-localized Evidence
              </div>
            )}
            <div className="confidence-badge" style={{ borderColor: confColor, color: confColor }}>
              <span className="conf-label">{confLabel} Confidence</span>
              <span className="conf-value">{(confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
        
        <div className="final-answer" style={{ fontSize: '16px', fontWeight: 'bold', margin: '15px 0' }}>
          {result.final_answer}
        </div>
        
        {result.caveats && result.caveats.length > 0 && (
          <div className="caveats-section" style={{ marginBottom: '15px', padding: '10px', background: '#fffbeb', borderLeft: '4px solid #f59e0b' }}>
            <strong style={{ fontSize: '14px', color: '#92400e' }}>Caveats:</strong>
            <ul style={{ margin: '5px 0 0 20px', fontSize: '13px', color: '#92400e' }}>
              {result.caveats.map((c: string, idx: number) => (
                <li key={idx}>{c}</li>
              ))}
            </ul>
          </div>
        )}
        
        <div className="claims-section" style={{ marginBottom: '20px' }}>
          <h3 style={{ fontSize: '14px', marginBottom: '8px', color: '#4b5563' }}>Evidence Claims (Click to highlight region)</h3>
          {evidenceNodes.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {evidenceNodes.map((node: any) => {
                const ev = node.data;
                const isSelected = selectedEvidenceId === ev.evidence_id;
                return (
                  <div 
                    key={ev.evidence_id}
                    onClick={() => setSelectedEvidenceId(isSelected ? null : ev.evidence_id)}
                    style={{ 
                      padding: '10px', 
                      border: isSelected ? '2px solid #3b82f6' : '1px solid #e5e7eb',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      background: isSelected ? '#eff6ff' : '#ffffff',
                      transition: 'all 0.2s'
                    }}
                  >
                    <div style={{ fontSize: '14px', fontWeight: '500', marginBottom: '4px' }}>{ev.claim}</div>
                    <div style={{ display: 'flex', gap: '10px', fontSize: '12px', color: '#6b7280' }}>
                      <span><strong>Model:</strong> {ev.source_model}</span>
                      <span><strong>Modality:</strong> {ev.modality}</span>
                      <span><strong>Conf:</strong> {(ev.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <ul style={{ listStyleType: 'disc', paddingLeft: '20px', fontSize: '14px', lineHeight: '1.5' }}>
              {result.claims?.map((c: string, idx: number) => (
                <li key={idx}>{c}</li>
              ))}
            </ul>
          )}
        </div>
        
        {!hasNonLocalized && (
          <div className="map-container-wrapper" style={{ marginTop: '20px' }}>
            <BeforeAfterViewer 
              evidenceGraph={evidenceGraph} 
              files={files} 
              jobId={jobId} 
              selectedEvidenceId={selectedEvidenceId} 
            />
          </div>
        )}
        
        <div className="exports-section" style={{ marginTop: '20px', paddingTop: '15px', borderTop: '1px solid #e5e7eb' }}>
          <h3 style={{ fontSize: '14px', marginBottom: '10px', color: '#4b5563' }}>Export Artifacts</h3>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <a href={`http://localhost:8000/api/jobs/${jobId}/export/pdf`} download className="export-link" style={{ padding: '6px 12px', background: '#f3f4f6', borderRadius: '4px', textDecoration: 'none', color: '#374151', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '5px' }}>
              📄 PDF Report
            </a>
            <a href={`http://localhost:8000/api/jobs/${jobId}/export/geojson`} download className="export-link" style={{ padding: '6px 12px', background: '#f3f4f6', borderRadius: '4px', textDecoration: 'none', color: '#374151', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '5px' }}>
              🗺️ GeoJSON
            </a>
            <a href={`http://localhost:8000/api/jobs/${jobId}/export/json_trace`} download className="export-link" style={{ padding: '6px 12px', background: '#f3f4f6', borderRadius: '4px', textDecoration: 'none', color: '#374151', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '5px' }}>
              🔍 JSON Trace
            </a>
            <a href={`http://localhost:8000/api/jobs/${jobId}/export/png`} download className="export-link" style={{ padding: '6px 12px', background: '#f3f4f6', borderRadius: '4px', textDecoration: 'none', color: '#374151', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '5px' }}>
              🖼️ Heatmap PNG
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="result-panel result-panel--processing">
      <div className="spinner"></div>
      <p>Processing ({status})...</p>
    </div>
  );
}
