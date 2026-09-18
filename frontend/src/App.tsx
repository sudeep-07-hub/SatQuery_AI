import { useEffect, useState } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import { STORAGE_KEYS } from './lib/chatStorage';
import HomePage from './pages/HomePage';
import AssistantPage from './pages/AssistantPage';

/** Old shared result links pointed at /?job=<id>; the workspace now lives at /assistant. */
function HomeOrLegacyJobLink() {
  const location = useLocation();
  if (new URLSearchParams(location.search).has('job')) {
    return <Navigate to={`/assistant${location.search}`} replace />;
  }
  return <HomePage />;
}

type Theme = 'light' | 'dark';

/** Remembered across navigations and reloads; falls back to the OS setting. Storage can throw in private mode. */
function initialTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.theme);
    if (stored === 'light' || stored === 'dark') return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem(STORAGE_KEYS.theme, theme);
    } catch {
      /* the page still works with an unremembered theme */
    }
  }, [theme]);

  return (
    <>
      <Navbar theme={theme} onToggleTheme={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))} />
      <Routes>
        <Route path="/" element={<HomeOrLegacyJobLink />} />
        <Route path="/assistant" element={<AssistantPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}
