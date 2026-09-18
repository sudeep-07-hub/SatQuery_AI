/**
 * The real MC1→MC8 stage sequence, shared by the Home panel and the Assistant's live progress view.
 *
 * `stages` lists the backend progress-trace stage names (job_manager.py) that belong to each step, so
 * progress is driven by what the backend actually reported — never by a timer.
 */
export interface PipelineStep {
  label: string;
  detail: string;
  /** One further true detail about the running system (see KNOWN_GAPS.md), revealed on demand. */
  more: string;
  stages: string[];
}

export const PIPELINE_STEPS: PipelineStep[] = [
  {
    label: 'Input qualification',
    detail: 'Reads the uploaded files and checks whether they can answer the question.',
    more: 'Format, coordinate system, footprint overlap and sensor type. A pair with no overlap is rejected here.',
    stages: ['QUEUED', 'MC1_VALIDATING'],
  },
  {
    label: 'Query understanding',
    detail: 'Turns the question into a task the pipeline can plan for.',
    more: 'Qwen3 4B (Q4_K_M, served by Ollama). If it is unavailable, deterministic Tool Registry rules take over and the answer says so.',
    stages: ['QUERY_INTELLIGENCE'],
  },
  {
    label: 'Tool registry',
    detail: 'Picks an engine that can actually run in this environment.',
    more: 'Each engine reports availability per job, so an engine that cannot run is never selected — ChangeMamba, for example, needs CUDA.',
    stages: ['TOOL_SELECTION', 'OBSERVATION_BINDING', 'WORKFLOW_PLANNING'],
  },
  {
    label: 'Specialist models',
    detail: 'Runs the selected engine on the uploaded pixels.',
    more: 'PaliGemma answers questions about a single image; change detection uses a classical SAR log-ratio / optical change vector analysis.',
    stages: ['AGENTIC_EXECUTION', 'EVIDENCE_NORMALIZATION'],
  },
  {
    label: 'Verification',
    detail: 'Checks each piece of evidence before it reaches the answer.',
    more: 'Evidence below the confidence threshold is rejected and listed as rejected. Confidence is raw model confidence, not calibrated.',
    stages: ['VERIFICATION'],
  },
  {
    label: 'Answer + audit',
    detail: 'Writes the answer from the verified evidence only.',
    more: 'Each answer carries its evidence, an auditable execution trace, and PDF, GeoJSON and JSON exports.',
    stages: ['ANSWER_SYNTHESIS', 'DONE'],
  },
];

/** Index of the step a backend stage belongs to, or -1 when the stage is not part of the sequence. */
export function stepIndexForStage(stage: string | null | undefined): number {
  if (!stage) return -1;
  return PIPELINE_STEPS.findIndex((step) => step.stages.includes(stage));
}
