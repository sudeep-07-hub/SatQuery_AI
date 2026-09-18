import type { ChatConfidence } from '../../lib/chatStorage';

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
  const calibrated = confidence.source !== 'model_confidence_uncalibrated';
  const pct = Math.round(confidence.value * 100);

  return (
    <div
      className={`confidence ${calibrated ? 'confidence--calibrated' : 'confidence--raw'}`}
      title={note ?? (calibrated ? 'Calibrated confidence' : 'Raw model confidence — MC6.2 calibration has not run')}
    >
      <span className="confidence__mark" aria-hidden="true" />
      <span className="confidence__label">
        {calibrated ? 'Calibrated confidence' : 'Model confidence (uncalibrated)'}
      </span>
      <span className="confidence__meter" role="img" aria-label={`${pct} percent`}>
        <span className="confidence__fill" style={{ inlineSize: `${pct}%` }} />
      </span>
      <span className="confidence__value">{(confidence.value).toFixed(2)}</span>
    </div>
  );
}
