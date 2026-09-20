import { useContext } from 'react';
import { I18nContext } from './I18nProvider';
import type { I18nContextValue, Translate } from './I18nProvider';

/** The whole i18n context — for components that also need to read or change the locale. */
export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) throw new Error('useI18n must be used inside <I18nProvider>');
  return context;
}

/** Just the translator, which is what most components want. */
export function useT(): Translate {
  return useI18n().t;
}
