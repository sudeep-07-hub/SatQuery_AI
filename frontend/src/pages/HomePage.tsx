import { Link } from 'react-router-dom';

// Each step names what the running pipeline actually does today (see KNOWN_GAPS.md for limits).
const PIPELINE_STEPS = [
  { label: 'Input qualification', detail: 'format · CRS · overlap · modality' },
  { label: 'Query understanding', detail: 'Qwen3 agentic controller' },
  { label: 'Tool registry', detail: 'selects available specialist models' },
  { label: 'Specialist models', detail: 'image Q&A · captioning · change detection' },
  { label: 'Verification', detail: 'per-evidence checks before answering' },
  { label: 'Answer + audit', detail: 'evidence-grounded response · execution trace' },
];

export default function HomePage() {
  return (
    <main className="home">
      <section className="home__hero">
        <div className="home__text">
          <h1 className="home__headline">Ask questions of satellite imagery. Get answers tied to evidence.</h1>
          <p className="home__lead">
            SatQuery AI is an agentic assistant for remote-sensing images: upload one or two optical or SAR
            images and ask a question in plain language.
          </p>
          <p className="home__lead">
            The controller interprets the query, selects specialist models from a tool registry, and returns an
            evidence-grounded response built only from what those models produced.
          </p>
          <p className="home__lead">
            Each answer includes its evidence items, an auditable execution trace, and model confidence reported
            as uncalibrated.
          </p>
          <Link to="/assistant" className="home__cta">
            Launch Assistant
          </Link>
        </div>

        <aside className="home__visual" aria-label="How a query is processed">
          <div className="home__visual-title">How a query runs</div>
          <ol className="home__steps">
            {PIPELINE_STEPS.map((step, idx) => (
              <li key={step.label} className="home__step">
                <span className="home__step-index">{String(idx + 1).padStart(2, '0')}</span>
                <span className="home__step-body">
                  <span className="home__step-label">{step.label}</span>
                  <span className="home__step-detail">{step.detail}</span>
                </span>
              </li>
            ))}
          </ol>
        </aside>
      </section>
    </main>
  );
}
