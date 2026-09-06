import React from 'react';
import BeforeAfterViewer from './BeforeAfterViewer';

interface ResultPanelProps {
  status: string;
  result: any;
  evidenceGraph: any;
  files: any[];
}

export default function ResultPanel({ status, result, evidenceGraph, files }: ResultPanelProps) {
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

    return (
      <div className="result-panel">
        <div className="result-header">
          <h2>Analysis Result</h2>
          <div className="confidence-badge" style={{ borderColor: confColor, color: confColor }}>
            <span className="conf-label">{confLabel} Confidence</span>
            <span className="conf-value">{(confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
        
        <div className="final-answer" style={{ fontSize: '16px', fontWeight: 'bold', margin: '15px 0' }}>
          {result.final_answer}
        </div>
        
        <div className="claims-section" style={{ marginBottom: '20px' }}>
          <h3 style={{ fontSize: '14px', marginBottom: '8px', color: '#4b5563' }}>Evidence Claims</h3>
          <ul style={{ listStyleType: 'disc', paddingLeft: '20px', fontSize: '14px', lineHeight: '1.5' }}>
            {result.claims?.map((c: string, idx: number) => (
              <li key={idx}>{c}</li>
            ))}
          </ul>
        </div>
        
        <div className="map-container-wrapper" style={{ marginTop: '20px' }}>
          <BeforeAfterViewer evidenceGraph={evidenceGraph} files={files} jobId={jobId} />
        </div>
        
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
