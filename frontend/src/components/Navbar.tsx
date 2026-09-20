import { useEffect, useState } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import LanguageSwitcher from './LanguageSwitcher';
import { useT } from '../i18n/useT';

const GITHUB_URL = 'https://github.com/sudeep-07-hub/SatQuery_AI';

interface NavbarProps {
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
}

/** Official GitHub mark (Octicons "mark-github"), coloured by the surrounding text token. */
function GitHubMark() {
  return (
    <svg viewBox="0 0 16 16" width="20" height="20" aria-hidden="true" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

export default function Navbar({ theme, onToggleTheme }: NavbarProps) {
  const t = useT();
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  // Close the mobile menu after navigating
  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  const linkClass = ({ isActive }: { isActive: boolean }) => `navbar__link ${isActive ? 'navbar__link--active' : ''}`;

  return (
    <header className="app-header navbar">
      <Link to="/" className="app-header__logo navbar__brand" aria-label={t('nav.brandAria')}>
        <div className="app-header__icon">🛰</div>
        <div className="app-header__title">SatQuery</div>
      </Link>

      <div className="navbar__controls">
        <button
          className="navbar__menu-toggle"
          aria-label={menuOpen ? t('nav.closeMenu') : t('nav.openMenu')}
          aria-expanded={menuOpen}
          aria-controls="navbar-links"
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? '✕' : '☰'}
        </button>

        <nav id="navbar-links" className={`navbar__links ${menuOpen ? 'navbar__links--open' : ''}`}>
          <NavLink to="/" end className={linkClass}>{t('nav.home')}</NavLink>
          <NavLink to="/assistant" className={linkClass}>{t('nav.assistant')}</NavLink>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="navbar__link navbar__github"
            aria-label={t('nav.githubAria')}
          >
            <GitHubMark />
            <span className="navbar__github-label">{t('nav.github')}</span>
          </a>
        </nav>

        <LanguageSwitcher />

        {/* Existing light/dark control, kept from the previous header (not a navigation item) */}
        <button className="theme-toggle" onClick={onToggleTheme} aria-label={t('nav.toggleTheme')}>
          {theme === 'light' ? '🌙' : '☀'}
        </button>
      </div>
    </header>
  );
}
