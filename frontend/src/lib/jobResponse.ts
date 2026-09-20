/**
 * Snapshot of a finished backend job, kept with the assistant chat turn.
 *
 * Field names are the backend's own (FastAPI `job_manager.py` result, GET /trace,
 * GET /structured_trace) — nothing here is renamed or invented. Large internals
 * (tool outputs, full evidence graph edges) are dropped to keep localStorage small.
 */

export const TERMINAL_STATES = [
  'DONE', 'FAILED', 'ABSTAIN', 'INSUFFICIENT_EVIDENCE', 'PRECONDITION_FAILED',
  'MODEL_UNAVAILABLE', 'INSUFFICIENT_OBSERVATIONS',
];

/** Evidence object as produced by MC5 normalisation (GET /structured_trace → evidence_objects[]). */
export interface EvidenceObject {
  evidence_id: string;
  claim: string;
  evidence_type?: string;
  spatial_region: { type: string; coordinates: unknown } | null;
  modality?: string;
  source_model?: string;
  confidence: number;
  processing_parameters?: Record<string, unknown>;
  [key: string]: unknown;
}

/** GET /api/jobs/{id}/trace entries (job_manager progress_trace). */
export interface ProgressTraceEntry {
  stage: string;
  message: string;
  timestamp: string;
  details?: unknown;
  error?: string;
  task_spec?: Record<string, unknown>;
  failed_constraints?: string[];
}

/** result.agent_state.execution_trace events (AgentController / RecoveryManager). */
export interface ExecutionTraceEvent {
  event_type: string;
  status: string;
  timestamp: string;
  call_id?: string;
  tool_id?: string;
  error?: string;
  reason?: string;
  [key: string]: unknown;
}

export interface ToolResultSummary {
  call_id: string;
  tool_id: string;
  status: string;
  error_information: string | null;
  execution_metadata: Record<string, unknown>;
}

export interface JobExecutionTrace {
  trace: ProgressTraceEntry[];
  execution_trace: ExecutionTraceEvent[];
  replan_history: Record<string, unknown>[];
  results: ToolResultSummary[];
  verification_result: {
    status: string;
    triggers_fired?: unknown[];
    rejected_claims?: EvidenceObject[];
    replan_count?: number;
  } | null;
  planner: Record<string, string | null> | null;
  tool_registry: { tool_id: string; name: string; available: boolean; reason: string }[] | null;
}

export interface JobResponse {
  final_answer: string;
  claims: string[];
  evidence_references: string[];
  confidence: number;
  confidence_note?: string;
  caveats: string[];
  execution_status?: string;
  verification_status?: string;
  change_statistics: Record<string, number | string | null> | null;
  change_overlay: {
    url: string;
    bounds_wgs84: [[number, number], [number, number]] | null;
    before_observation?: string;
    after_observation?: string;
  } | null;
  observations: Record<string, {
    filename: string;
    georeferenced: boolean;
    bounds_wgs84: [[number, number], [number, number]] | null;
    preview_url: string;
  }> | null;
  failed?: { reason: string }[];
  evidence_objects: EvidenceObject[];
  /** Language the question was asked in; "en" unless the user switched the interface language. */
  query_language?: string;
  /**
   * Present only when the backend actually translated the query. Both strings are kept so a
   * reviewer can see the exact text the pipeline reasoned over, rather than trusting that the
   * translation was faithful.
   */
  input_translation?: {
    from: string;
    to: string;
    engine: string;
    original: string;
    translated: string;
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Raw = any;

export function snapshotJob(result: Raw, progressTrace: Raw[], structuredTrace: Raw): { response: JobResponse; executionTrace: JobExecutionTrace } {
  const r = result ?? {};
  const state = r.agent_state ?? {};
  const response: JobResponse = {
    final_answer: typeof r.final_answer === 'string' ? r.final_answer : '',
    claims: Array.isArray(r.claims) ? r.claims : [],
    evidence_references: Array.isArray(r.evidence_references) ? r.evidence_references : [],
    confidence: typeof r.confidence === 'number' ? r.confidence : 0,
    confidence_note: r.confidence_note,
    caveats: Array.isArray(r.caveats) ? r.caveats : [],
    execution_status: r.execution_status,
    verification_status: r.verification_status,
    change_statistics: r.change_statistics ?? null,
    change_overlay: r.change_overlay ?? null,
    observations: r.observations ?? null,
    failed: Array.isArray(r.failed) ? r.failed : undefined,
    evidence_objects: Array.isArray(structuredTrace?.evidence_objects) ? structuredTrace.evidence_objects : [],
    query_language: typeof r.query_language === 'string' ? r.query_language : undefined,
    input_translation: r.input_translation ?? undefined,
  };
  const executionTrace: JobExecutionTrace = {
    // Tracebacks are server diagnostics, not user-facing: never persisted or shown
    trace: (progressTrace ?? []).map((entry: Raw) => {
      const { traceback: _traceback, ...rest } = entry ?? {};
      return rest;
    }),
    execution_trace: Array.isArray(state.execution_trace) ? state.execution_trace : [],
    replan_history: Array.isArray(state.replan_history) ? state.replan_history : [],
    results: Object.values(state.results ?? {}).map((res: Raw) => ({
      call_id: res.call_id,
      tool_id: res.tool_id,
      status: res.status,
      error_information: res.error_information ?? null,
      execution_metadata: res.execution_metadata ?? {},
    })),
    verification_result: r.verification_result ?? null,
    planner: r.planner ?? null,
    tool_registry: r.tool_registry ?? null,
  };
  return { response, executionTrace };
}

/** Spatial evidence exists only when the backend produced a change overlay or a non-null evidence region. */
export function hasSpatialEvidence(response: JobResponse | undefined): boolean {
  if (!response || !response.observations || Object.keys(response.observations).length === 0) return false;
  return !!response.change_overlay || response.evidence_objects.some((ev) => ev.spatial_region !== null);
}
