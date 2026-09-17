import { useState } from 'react';
import { Link } from 'react-router-dom';

// The six steps are the real MC1→MC8 order. `more` reveals one further detail that is true of the
// running system today (see KNOWN_GAPS.md) — including which engines are not active.
const PIPELINE_STEPS = [
  {
    label: 'Input qualification',
    detail: 'Reads the uploaded files and checks whether they can answer the question.',
    more: 'Format, coordinate system, footprint overlap and sensor type. A pair with no overlap is rejected here.',
  },
  {
    label: 'Query understanding',
    detail: 'Turns the question into a task the pipeline can plan for.',
    more: 'Qwen3 4B (Q4_K_M, served by Ollama). If it is unavailable, deterministic Tool Registry rules take over and the answer says so.',
  },
  {
    label: 'Tool registry',
    detail: 'Picks an engine that can actually run in this environment.',
    more: 'Each engine reports availability per job, so an engine that cannot run is never selected — ChangeMamba, for example, needs CUDA.',
  },
  {
    label: 'Specialist models',
    detail: 'Runs the selected engine on the uploaded pixels.',
    more: 'PaliGemma answers questions about a single image; change detection uses a classical SAR log-ratio / optical change vector analysis.',
  },
  {
    label: 'Verification',
    detail: 'Checks each piece of evidence before it reaches the answer.',
    more: 'Evidence below the confidence threshold is rejected and listed as rejected. Confidence is raw model confidence, not calibrated.',
  },
  {
    label: 'Answer + audit',
    detail: 'Writes the answer from the verified evidence only.',
    more: 'Each answer carries its evidence, an auditable execution trace, and PDF, GeoJSON and JSON exports.',
  },
];

/** Decorative orbit behind the hero text: a satellite tracing an ellipse, frozen under reduced motion. */
function OrbitBackdrop() {
  return (
    <div className="orbit" aria-hidden="true">
      <svg className="orbit__svg" viewBox="0 0 600 600" role="presentation" focusable="false">
        <ellipse className="orbit__ring" cx="300" cy="300" rx="250" ry="150" transform="rotate(-18 300 300)" />
        <ellipse className="orbit__ring orbit__ring--inner" cx="300" cy="300" rx="170" ry="100" transform="rotate(-18 300 300)" />
        <circle className="orbit__planet" cx="300" cy="300" r="62" />
      </svg>
      <span className="orbit__satellite" />
    </div>
  );
}

export default function HomePage() {
  const [openStep, setOpenStep] = useState<number | null>(null);

  return (
    <main className="home">
      <section className="home__hero">
        <div className="home__text">
          <OrbitBackdrop />
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
            <span className="home__cta-arrow" aria-hidden="true">→</span>
          </Link>
        </div>

        <aside className="home__visual" aria-label="How a query is processed">
          <p className="home__visual-lead">How a query runs</p>
          <ol className="home__steps">
            {PIPELINE_STEPS.map((step, idx) => {
              const open = openStep === idx;
              return (
                <li key={step.label} className={`home__step ${open ? 'home__step--open' : ''}`}>
                  <button
                    type="button"
                    className="home__step-button"
                    aria-expanded={open}
                    onClick={() => setOpenStep(open ? null : idx)}
                  >
                    <span className="home__step-index">{String(idx + 1).padStart(2, '0')}</span>
                    <span className="home__step-body">
                      <span className="home__step-label">{step.label}</span>
                      <span className="home__step-detail">{step.detail}</span>
                      <span className="home__step-more"><span>{step.more}</span></span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
        </aside>
      </section>
    </main>
  );
}
