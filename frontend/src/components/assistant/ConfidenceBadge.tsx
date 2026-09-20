import type { ChatConfidence } from '../../lib/chatStorage';
import { useT } from '../../i18n/useT';

interface ConfidenceBadgeProps {
  confidence: ChatConfidence;
  /** result.confidence_note from the backend, shown as the tooltip. */
  note?: string;
}

/**
 * Confidence gauge. The two sources are distinguishable at a glance, not only by wording:
 * raw model confidence gets a dashed outline and an open marker; calibrated confidence (only
 * possible once MC6.2 exists and reports calibrated_confidence) gets a solid outline and filled mark.
 */
export default function ConfidenceBadge({ confidence, note }: ConfidenceBadgeProps) {
  const t = useT();
  const calibrated = confidence.source !== 'model_confidence_uncalibrated';
  const pct = Math.round(confidence.value * 100);

  return (
    <div
      className={`confidence ${calibrated ? 'confidence--calibrated' : 'confidence--raw'}`}
      title={note ?? (calibrated ? t('confidence.calibrated') : t('confidence.rawTitle'))}
    >
      <span className="confidence__mark" aria-hidden="true" />
      <span className="confidence__label">
        {calibrated ? t('confidence.calibrated') : t('confidence.uncalibrated')}
      </span>
      <span className="confidence__meter" role="img" aria-label={t('confidence.percentAria', { pct })}>
        <span className="confidence__fill" style={{ inlineSize: `${pct}%` }} />
      </span>
      <span className="confidence__value">{(confidence.value).toFixed(2)}</span>
    </div>
  );
}
