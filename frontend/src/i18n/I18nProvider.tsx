import { createContext, useCallback, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { STORAGE_KEYS } from '../lib/chatStorage';
import { en } from './en';
import { hi } from './hi';
import { kn } from './kn';
import { te } from './te';
import { ta } from './ta';
import { isLocaleCode, LOCALE_FONT } from './types';
import type { Locale, LocaleCode, TranslationKey } from './types';

const CATALOGS: Record<LocaleCode, Locale> = { en, hi, kn, te, ta };

export type TranslateVars = Record<string, string | number>;
export type Translate = (key: TranslationKey, vars?: TranslateVars) => string;

export interface I18nContextValue {
  locale: LocaleCode;
  setLocale: (locale: LocaleCode) => void;
  t: Translate;
}

export const I18nContext = createContext<I18nContextValue | null>(null);

/** `{name}` substitution. An unknown placeholder is left verbatim rather than blanked. */
function interpolate(template: string, vars?: TranslateVars): string {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (match, name: string) =>
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match);
}

/** Stored choice → the browser's preferred language → English. Storage can throw in private mode. */
function initialLocale(): LocaleCode {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.lang);
    if (isLocaleCode(stored)) return stored;
  } catch {
    /* fall through to the browser's preference */
  }
  for (const tag of navigator.languages ?? [navigator.language]) {
    // Match the primary subtag only: "hi-IN" and "hi" both select Hindi.
    const base = tag.toLowerCase().split('-')[0];
    if (isLocaleCode(base)) return base;
  }
  return 'en';
}

/**
 * Adds the stylesheet for a locale's script, once. English needs nothing — the Latin families are
 * already in index.html — so an English visitor never downloads an Indic font.
 */
function ensureScriptFont(locale: LocaleCode): void {
  const font = LOCALE_FONT[locale];
  if (!font) return;
  const id = `satquery-font-${locale}`;
  if (document.getElementById(id)) return;
  const link = document.createElement('link');
  link.id = id;
  link.rel = 'stylesheet';
  link.href = font.href;
  document.head.appendChild(link);
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<LocaleCode>(initialLocale);

  useEffect(() => {
    document.documentElement.lang = locale;
    ensureScriptFont(locale);
    try {
      localStorage.setItem(STORAGE_KEYS.lang, locale);
    } catch {
      /* the page still works with an unremembered language */
    }
  }, [locale]);

  const t = useCallback<Translate>(
    (key, vars) => interpolate(CATALOGS[locale][key], vars),
    [locale],
  );

  const value = useMemo<I18nContextValue>(() => ({ locale, setLocale, t }), [locale, t]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}
