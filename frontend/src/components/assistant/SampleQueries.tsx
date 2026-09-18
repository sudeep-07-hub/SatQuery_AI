import { SAMPLE_QUERIES, type SampleQuery } from '../../lib/samples';

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
  return (
    <div className="samples">
      <p className="samples__lead">
        Sample queries — each one uploads the bundled imagery shown on the card and runs the pipeline live.
      </p>
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
              title={unavailable ? tool?.reason : `Runs: ${sample.query}`}
            >
              <span className="sample__title">{sample.title}</span>
              <span className="sample__query">“{sample.query}”</span>
              <span className="sample__meta">
                <span className="sample__count">{sample.files.length === 2 ? '2 images' : '1 image'}</span>
                <span className="sample__source">{sample.source}</span>
              </span>
              {unavailable && (
                <span className="sample__unavailable">Not available on the connected backend</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
