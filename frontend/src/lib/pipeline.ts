/**
 * The real MC1→MC8 stage sequence, shared by the Home panel and the Assistant's live progress view.
 *
 * `stages` lists the backend progress-trace stage names (job_manager.py) that belong to each step, so
 * progress is driven by what the backend actually reported — never by a timer.
 */
import type { TranslationKey } from '../i18n/types';

export interface PipelineStep {
  labelKey: TranslationKey;
  detailKey: TranslationKey;
  /** One further true detail about the running system (see KNOWN_GAPS.md), revealed on demand. */
  moreKey: TranslationKey;
  /** Backend progress-trace stage names — identifiers, never translated. */
  stages: string[];
}

export const PIPELINE_STEPS: PipelineStep[] = [
  {
    labelKey: 'pipeline.qualify.label',
    detailKey: 'pipeline.qualify.detail',
    moreKey: 'pipeline.qualify.more',
    stages: ['QUEUED', 'MC1_VALIDATING'],
  },
  {
    labelKey: 'pipeline.query.label',
    detailKey: 'pipeline.query.detail',
    moreKey: 'pipeline.query.more',
    stages: ['QUERY_INTELLIGENCE'],
  },
  {
    labelKey: 'pipeline.registry.label',
    detailKey: 'pipeline.registry.detail',
    moreKey: 'pipeline.registry.more',
    stages: ['TOOL_SELECTION', 'OBSERVATION_BINDING', 'WORKFLOW_PLANNING'],
  },
  {
    labelKey: 'pipeline.models.label',
    detailKey: 'pipeline.models.detail',
    moreKey: 'pipeline.models.more',
    stages: ['AGENTIC_EXECUTION', 'EVIDENCE_NORMALIZATION'],
  },
  {
    labelKey: 'pipeline.verify.label',
    detailKey: 'pipeline.verify.detail',
    moreKey: 'pipeline.verify.more',
    stages: ['VERIFICATION'],
  },
  {
    labelKey: 'pipeline.answer.label',
    detailKey: 'pipeline.answer.detail',
    moreKey: 'pipeline.answer.more',
    stages: ['ANSWER_SYNTHESIS', 'DONE'],
  },
];

/** Index of the step a backend stage belongs to, or -1 when the stage is not part of the sequence. */
export function stepIndexForStage(stage: string | null | undefined): number {
  if (!stage) return -1;
  return PIPELINE_STEPS.findIndex((step) => step.stages.includes(stage));
}
