import { SAMPLE_QUERIES, type SampleQuery } from '../../lib/samples';
import { useT } from '../../i18n/useT';

interface SampleQueriesProps {
  /** Tool availability from GET /api/system; a sample whose engine is unavailable is disabled with the reason. */
  tools: { tool_id: string; available: boolean; reason: string }[] | null;
  busy: boolean;
  onRun: (sample: SampleQuery) => void;
}

/**
 * Sample queries for the empty state. Each card ships real imagery with the site and runs the real
 * pipeline when clicked — no recorded answers.
 */
export default function SampleQueries({ tools, busy, onRun }: SampleQueriesProps) {
  const t = useT();
  return (
    <div className="samples">
      <p className="samples__lead">{t('sample.lead')}</p>
      <div className="samples__grid">
        {SAMPLE_QUERIES.map((sample) => {
          const tool = tools?.find((t) => t.tool_id === sample.requiresTool);
          const unavailable = tools ? !tool?.available : false;
          return (
            <button
              key={sample.id}
              type="button"
              className={`sample ${unavailable ? 'sample--unavailable' : ''}`}
              disabled={busy || unavailable}
              onClick={() => onRun(sample)}
              title={unavailable ? tool?.reason : t('sample.runs', { query: sample.query })}
            >
              <span className="sample__title">{t(sample.titleKey)}</span>
              <span className="sample__query">“{sample.query}”</span>
              <span className="sample__meta">
                <span className="sample__count">{t(sample.files.length === 2 ? 'sample.twoImages' : 'sample.oneImage')}</span>
                <span className="sample__source">{sample.source}</span>
              </span>
              {unavailable && (
                <span className="sample__unavailable">{t('sample.unavailable')}</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
