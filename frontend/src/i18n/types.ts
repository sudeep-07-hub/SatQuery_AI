import type { en } from './en';

/**
 * `en.ts` is the source of truth. Every other locale is typed `Locale`, so a missing key is a
 * compile error and an extra key is flagged by excess-property checking — there is no runtime
 * key fallback anywhere in this module.
 */
export type TranslationKey = keyof typeof en;
export type Locale = Record<TranslationKey, string>;

export const LOCALE_CODES = ['en', 'hi', 'kn', 'te', 'ta'] as const;
export type LocaleCode = (typeof LOCALE_CODES)[number];

/**
 * Each language named in its own script. Not a flag and not a country code: flags map countries,
 * and several of these languages are spoken across more than one.
 */
export const LOCALE_ENDONYM: Record<LocaleCode, string> = {
  en: 'English',
  hi: 'हिन्दी',
  kn: 'ಕನ್ನಡ',
  te: 'తెలుగు',
  ta: 'தமிழ்',
};

/**
 * The webfont each locale needs, and the family name the stylesheet switches to. IBM Plex Sans
 * carries no Indic glyphs, so without this four of the five languages render as tofu. English
 * needs nothing extra — the Latin families are already in index.html.
 *
 * IBM Plex has a Devanagari cut, which keeps Hindi in the app's own type voice. It has no Kannada,
 * Telugu or Tamil cut (checked against Google Fonts: those families 404), so those three use Noto
 * Sans, which is the closest available match in tone and is designed to harmonise across scripts.
 */
export interface ScriptFont {
  family: string;
  href: string;
}

export const LOCALE_FONT: Record<LocaleCode, ScriptFont | null> = {
  en: null,
  hi: {
    family: 'IBM Plex Sans Devanagari',
    href: 'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Devanagari:wght@400;600&display=swap',
  },
  kn: {
    family: 'Noto Sans Kannada',
    href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+Kannada:wght@400;600&display=swap',
  },
  te: {
    family: 'Noto Sans Telugu',
    href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;600&display=swap',
  },
  ta: {
    family: 'Noto Sans Tamil',
    href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil:wght@400;600&display=swap',
  },
};

export function isLocaleCode(value: unknown): value is LocaleCode {
  return typeof value === 'string' && (LOCALE_CODES as readonly string[]).includes(value);
}
