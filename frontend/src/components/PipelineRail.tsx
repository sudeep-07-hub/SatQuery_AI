import { PIPELINE_STEPS } from '../lib/pipeline';
import { useT } from '../i18n/useT';

interface PipelineRailProps {
  /** "home": each step can be expanded for one more detail. "progress": a live job's stage is highlighted. */
  variant: 'home' | 'progress';
  openStep?: number | null;
  onToggleStep?: (index: number) => void;
  /** Index of the step the backend is currently reporting (progress variant). */
  activeStep?: number;
  /** Stage name as reported by the backend, shown verbatim next to the active step. */
  activeStageLabel?: string | null;
}

/**
 * The six MC1→MC8 steps on a connecting rail. Shared by the Home page and the Assistant's live
 * progress view so both read as one system.
 */
export default function PipelineRail({
  variant, openStep = null, onToggleStep, activeStep = -1, activeStageLabel,
}: PipelineRailProps) {
  const t = useT();
  return (
    <ol className={`rail rail--${variant}`}>
      {PIPELINE_STEPS.map((step, idx) => {
        const open = openStep === idx;
        const state = variant === 'progress'
          ? (idx < activeStep ? 'done' : idx === activeStep ? 'active' : 'pending')
          : undefined;

        const body = (
          <span className="rail__body">
            <span className="rail__label">{t(step.labelKey)}</span>
            <span className="rail__detail">{t(step.detailKey)}</span>
            {variant === 'home' && (
              <span className="rail__more"><span>{t(step.moreKey)}</span></span>
            )}
            {variant === 'progress' && state === 'active' && activeStageLabel && (
              <span className="rail__stage">{activeStageLabel}</span>
            )}
          </span>
        );

        return (
          <li
            key={step.labelKey}
            className={`rail__step ${open ? 'rail__step--open' : ''} ${state ? `rail__step--${state}` : ''}`}
            aria-current={state === 'active' ? 'step' : undefined}
          >
            {variant === 'home' ? (
              <button type="button" className="rail__button" aria-expanded={open} onClick={() => onToggleStep?.(idx)}>
                <span className="rail__index">{String(idx + 1).padStart(2, '0')}</span>
                {body}
              </button>
            ) : (
              <div className="rail__button rail__button--static">
                <span className="rail__index">{String(idx + 1).padStart(2, '0')}</span>
                {body}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
