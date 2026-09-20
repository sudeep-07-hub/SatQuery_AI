import { useCallback, useEffect, useId, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { useI18n } from '../i18n/useT';
import { LOCALE_CODES, LOCALE_ENDONYM } from '../i18n/types';
import type { LocaleCode } from '../i18n/types';

/**
 * Language switcher — a custom listbox, not a native <select>, so it can carry the app's own
 * surface, border, radius and accent tokens.
 *
 * Each language is named in its own script. Options for scripts whose webfont is not loaded fall
 * back to the operating system's font for that script: loading all four Indic webfonts just to
 * draw this menu would defeat the per-script loading the provider does, and would mean an English
 * visitor downloading fonts they never see text in.
 */
export default function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(() => LOCALE_CODES.indexOf(locale));
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const baseId = useId();
  const optionId = (code: LocaleCode) => `${baseId}-${code}`;

  const close = useCallback((returnFocus: boolean) => {
    setOpen(false);
    if (returnFocus) triggerRef.current?.focus();
  }, []);

  // Focus moves into the list when it opens, so arrow keys act on the options.
  useEffect(() => {
    if (open) listRef.current?.focus();
  }, [open]);

  // Click outside and scroll both dismiss. Scroll is captured because the page's scrolling
  // container is not window on the Assistant route.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) close(false);
    };
    const onScroll = () => close(false);
    document.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('scroll', onScroll, { capture: true, passive: true });
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('scroll', onScroll, { capture: true });
    };
  }, [open, close]);

  const openAt = (index: number) => {
    setActiveIndex(index);
    setOpen(true);
  };

  const onTriggerKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const current = LOCALE_CODES.indexOf(locale);
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'ArrowDown') {
      event.preventDefault();
      openAt(current);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      openAt(LOCALE_CODES.length - 1);
    }
  };

  const onListKeyDown = (event: KeyboardEvent<HTMLUListElement>) => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, LOCALE_CODES.length - 1));
        break;
      case 'ArrowUp':
        event.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
        break;
      case 'Home':
        event.preventDefault();
        setActiveIndex(0);
        break;
      case 'End':
        event.preventDefault();
        setActiveIndex(LOCALE_CODES.length - 1);
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        setLocale(LOCALE_CODES[activeIndex]);
        close(true);
        break;
      case 'Escape':
        event.preventDefault();
        close(true);
        break;
      case 'Tab':
        close(false);
        break;
      default:
        break;
    }
  };

  return (
    <div className="lang-switcher" ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="lang-switcher__trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={t('nav.languageChoose')}
        onClick={() => (open ? close(false) : openAt(LOCALE_CODES.indexOf(locale)))}
        onKeyDown={onTriggerKeyDown}
      >
        <span className="lang-switcher__current">{LOCALE_ENDONYM[locale]}</span>
        <span className="lang-switcher__caret" aria-hidden="true">▾</span>
      </button>

      {open && (
        <ul
          ref={listRef}
          className="lang-switcher__panel"
          role="listbox"
          tabIndex={-1}
          aria-label={t('nav.language')}
          aria-activedescendant={optionId(LOCALE_CODES[activeIndex])}
          onKeyDown={onListKeyDown}
        >
          {LOCALE_CODES.map((code, index) => (
            <li
              key={code}
              id={optionId(code)}
              role="option"
              aria-selected={code === locale}
              className={
                'lang-switcher__option'
                + (code === locale ? ' lang-switcher__option--selected' : '')
                + (index === activeIndex ? ' lang-switcher__option--active' : '')
              }
              onPointerEnter={() => setActiveIndex(index)}
              onClick={() => {
                setLocale(code);
                close(true);
              }}
            >
              {LOCALE_ENDONYM[code]}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
