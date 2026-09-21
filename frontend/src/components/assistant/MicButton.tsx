import { useT } from '../../i18n/useT';
import type { SpeechState } from '../../lib/speech';

interface MicButtonProps {
  supported: boolean;
  languageUnsupported: boolean;
  /** Name of the current interface language, in its own script, for the unsupported message. */
  languageName: string;
  state: SpeechState;
  /** Disabled for a reason outside speech, e.g. a query is already running. */
  busy: boolean;
  /** Hard cap, surfaced before the user starts rather than as a surprise cut-off. */
  maxSeconds: number;
  onStart: () => void;
  onStop: () => void;
}

function MicGlyph({ listening }: { listening: boolean }) {
  return (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"
         fill={listening ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="1.3">
      <rect x="5.6" y="1.6" width="4.8" height="8" rx="2.4" />
      <path d="M3.2 7.2v.8a4.8 4.8 0 0 0 9.6 0v-.8" strokeLinecap="round" fill="none" />
      <path d="M8 12.8v1.6" strokeLinecap="round" fill="none" />
    </svg>
  );
}

/**
 * Dictation toggle, in the in-field icon row established in Phase 1 — same box, gap and inset as
 * any other icon there, no special sizing.
 *
 * Presentational: the recogniser lives in useSpeechRecognition, so the composer owns one copy of
 * the state and can render the error message in its own error slot rather than inside this row.
 * When speech is unavailable the button stays visible and disabled with the real reason in its
 * tooltip, the same way an unavailable sample card reports its backend reason.
 */
export default function MicButton({
  supported, languageUnsupported, languageName, state, busy, maxSeconds, onStart, onStop,
}: MicButtonProps) {
  const t = useT();
  const listening = state === 'listening' || state === 'requesting-permission';
  const unavailable = !supported || languageUnsupported;
  const disabled = unavailable || busy || state === 'finalising';

  const reason = !supported
    ? t('mic.unsupported')
    : languageUnsupported
      ? t('mic.langUnsupported', { lang: languageName })
      : listening
        ? t('mic.stop')
        : `${t('mic.start')} — ${t('mic.limitNote', { seconds: maxSeconds })}`;

  return (
    <>
      <button
        type="button"
        className={`chat-composer__icon-btn mic${listening ? ' mic--listening' : ''}`}
        onClick={listening ? onStop : onStart}
        disabled={disabled}
        aria-pressed={listening}
        aria-label={listening ? t('mic.stop') : t('mic.start')}
        title={reason}
      >
        <MicGlyph listening={listening} />
      </button>
      {/* Announced without stealing focus, so a screen-reader user knows the mic opened or closed. */}
      <span className="sr-only" role="status" aria-live="polite">
        {state === 'listening' ? t('mic.listening') : state === 'idle' ? t('mic.stopped') : ''}
      </span>
    </>
  );
}
