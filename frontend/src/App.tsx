import { useEffect, useState } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
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

export default function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
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
