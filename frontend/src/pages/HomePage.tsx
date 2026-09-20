import { useT } from '../i18n/useT';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import PipelineRail from '../components/PipelineRail';

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
  const t = useT();
  const [openStep, setOpenStep] = useState<number | null>(null);

  return (
    <main className="home">
      <section className="home__hero">
        <div className="home__text">
          <OrbitBackdrop />
          <h1 className="home__headline">{t('home.tagline')}</h1>
          <p className="home__lead">{t('home.lead1')}</p>
          <p className="home__lead">{t('home.lead2')}</p>
          <p className="home__lead">{t('home.lead3')}</p>
          <Link to="/assistant" className="home__cta">
            {t('home.launch')}
            <span className="home__cta-arrow" aria-hidden="true">→</span>
          </Link>
        </div>

        <aside className="home__visual" aria-label={t('home.pipelineAria')}>
          <p className="home__visual-lead">{t('home.pipelineTitle')}</p>
          <PipelineRail
            variant="home"
            openStep={openStep}
            onToggleStep={(idx) => setOpenStep(openStep === idx ? null : idx)}
          />
        </aside>
      </section>
    </main>
  );
}
