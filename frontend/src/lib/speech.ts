import { useCallback, useEffect, useRef, useState } from 'react';
import type { LocaleCode } from '../i18n/types';

/**
 * Browser-native speech recognition for the composer.
 *
 * Nothing is sent to the SatQuery backend: this is the Web Speech API, which in Chrome relays
 * audio to Google's speech service. There is deliberately no server-side transcription — the
 * hosted demo runs on 512 MB and could not load Whisper, and pretending otherwise would break the
 * honest-capability story the rest of the app keeps.
 */

/** Minimal shape of the vendor-prefixed API; it is not in the TS DOM lib. */
interface SpeechRecognitionAlternativeLike { transcript: string }
interface SpeechRecognitionResultLike {
  readonly isFinal: boolean;
  readonly length: number;
  [index: number]: SpeechRecognitionAlternativeLike;
}
interface SpeechRecognitionResultListLike {
  readonly length: number;
  [index: number]: SpeechRecognitionResultLike;
}
interface SpeechRecognitionEventLike {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultListLike;
}
interface SpeechRecognitionErrorEventLike { readonly error: string }
interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  onaudiostart: (() => void) | null;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function recognitionConstructor(): SpeechRecognitionCtor | null {
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/** Recognition locale per interface language. India variants: this is an Indian EO dataset. */
export const RECOGNITION_LANG: Record<LocaleCode, string> = {
  en: 'en-IN',
  hi: 'hi-IN',
  kn: 'kn-IN',
  te: 'te-IN',
  ta: 'ta-IN',
};

/** Dictation stops itself here; a runaway recogniser should not hold the microphone open. */
export const MAX_DICTATION_SECONDS = 30;

export type SpeechState = 'idle' | 'requesting-permission' | 'listening' | 'finalising' | 'error';

/**
 * Languages the speech service rejected at runtime.
 *
 * The Web Speech API exposes no list of supported languages — the only way to find out is to try
 * and read the `language-not-supported` error. Remembering it here lets the mic disable itself for
 * that language with the real reason instead of failing the same way on every click.
 */
const unsupportedLanguages = new Set<string>();

export interface SpeechRecognitionHook {
  /** False when the browser has no Web Speech API at all (Firefox, some Safari builds). */
  supported: boolean;
  /** True when the API exists but the speech service rejected this particular language. */
  languageUnsupported: boolean;
  state: SpeechState;
  /** Error code as reported by the browser, e.g. "not-allowed"; null when there is none. */
  errorCode: string | null;
  /** True once permission was denied, so the UI can stop re-prompting on every click. */
  permissionDenied: boolean;
  start: () => void;
  stop: () => void;
  dismissError: () => void;
}

export function useSpeechRecognition(
  locale: LocaleCode,
  onFinalTranscript: (text: string) => void,
  onInterimTranscript: (text: string) => void,
): SpeechRecognitionHook {
  const Ctor = recognitionConstructor();
  const supported = Ctor !== null;
  const lang = RECOGNITION_LANG[locale];

  const [state, setState] = useState<SpeechState>('idle');
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [languageUnsupported, setLanguageUnsupported] = useState(() => unsupportedLanguages.has(lang));

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const timerRef = useRef<number | null>(null);
  // Kept in refs so the recogniser's callbacks never close over a stale render.
  const finalRef = useRef(onFinalTranscript);
  const interimRef = useRef(onInterimTranscript);
  finalRef.current = onFinalTranscript;
  interimRef.current = onInterimTranscript;

  useEffect(() => {
    setLanguageUnsupported(unsupportedLanguages.has(lang));
  }, [lang]);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    clearTimer();
    const recognition = recognitionRef.current;
    if (!recognition) {
      setState('idle');
      return;
    }
    setState('finalising');
    // stop() lets a final result arrive; abort() would discard what was already spoken.
    recognition.stop();
  }, [clearTimer]);

  const start = useCallback(() => {
    if (!Ctor || unsupportedLanguages.has(lang)) return;
    if (recognitionRef.current) return;

    const recognition = new Ctor();
    recognition.lang = lang;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onaudiostart = () => setState('listening');

    recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result[0]?.transcript ?? '';
        if (result.isFinal) {
          finalRef.current(text);
        } else {
          interim += text;
        }
      }
      interimRef.current(interim);
    };

    recognition.onerror = (event) => {
      const code = event.error;
      setErrorCode(code);
      setState('error');
      if (code === 'not-allowed' || code === 'service-not-allowed') setPermissionDenied(true);
      if (code === 'language-not-supported') {
        unsupportedLanguages.add(lang);
        setLanguageUnsupported(true);
      }
    };

    recognition.onend = () => {
      clearTimer();
      recognitionRef.current = null;
      interimRef.current('');
      setState((current) => (current === 'error' ? 'error' : 'idle'));
    };

    recognitionRef.current = recognition;
    setErrorCode(null);
    setState('requesting-permission');
    try {
      recognition.start();
    } catch {
      // start() throws if called twice in a row; treat it as "already running".
      recognitionRef.current = null;
      setState('idle');
      return;
    }
    timerRef.current = window.setTimeout(stop, MAX_DICTATION_SECONDS * 1000);
  }, [Ctor, lang, clearTimer, stop]);

  // Stop the recogniser if the composer unmounts mid-dictation.
  useEffect(() => () => {
    clearTimer();
    recognitionRef.current?.abort();
    recognitionRef.current = null;
  }, [clearTimer]);

  const dismissError = useCallback(() => {
    setErrorCode(null);
    setState('idle');
  }, []);

  return { supported, languageUnsupported, state, errorCode, permissionDenied, start, stop, dismissError };
}
