/**
 * Chat history persistence — client-side only (localStorage).
 *
 * DEMO-SCOPE SHIM: history lives in this browser only. A backend-persisted session API is the
 * likely future upgrade; swap loadSessions/saveSessions for API calls when one exists.
 *
 * Every read/write is wrapped in try/catch: a missing, corrupted, or over-quota store degrades
 * to empty history and never crashes the page.
 */

export const STORAGE_KEYS = {
  sidebarCollapsed: 'satquery.sidebar.collapsed',
  sessions: 'satquery.chat.sessions',
} as const;

export const MAX_SESSIONS = 50;
const TITLE_MAX_CHARS = 60;
export const NEW_CHAT_TITLE = 'New chat';

export interface ChatAttachment {
  name: string;
  size: number;
  type: string;
}

/**
 * Confidence as reported by the backend result (`result.confidence`). MC6.2 calibration is not
 * implemented, so the only valid source today is raw, uncalibrated model confidence.
 */
export interface ChatConfidence {
  value: number;
  source: 'model_confidence_uncalibrated';
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  attachments?: ChatAttachment[];
  executionTrace?: unknown;
  confidence?: ChatConfidence;
  timestamp: string;
  /** Backend job this message belongs to (lets the full result be re-fetched while the server still holds it). */
  jobId?: string;
  /** Terminal job status, e.g. DONE, ABSTAIN, INSUFFICIENT_EVIDENCE. */
  status?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

function isMessage(value: unknown): value is ChatMessage {
  const m = value as ChatMessage;
  return !!m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && typeof m.timestamp === 'string';
}

function isSession(value: unknown): value is ChatSession {
  const s = value as ChatSession;
  return !!s && typeof s.id === 'string' && typeof s.title === 'string'
    && typeof s.createdAt === 'string' && typeof s.updatedAt === 'string'
    && Array.isArray(s.messages) && s.messages.every(isMessage);
}

export function sortByRecent(sessions: ChatSession[]): ChatSession[] {
  return [...sessions].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

export function loadSessions(): ChatSession[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEYS.sessions);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Drop malformed entries individually rather than discarding the whole history
    return sortByRecent(parsed.filter(isSession)).slice(0, MAX_SESSIONS);
  } catch {
    return [];
  }
}

export function saveSessions(sessions: ChatSession[]): void {
  let toStore = sortByRecent(sessions).slice(0, MAX_SESSIONS);
  try {
    while (toStore.length > 0) {
      try {
        window.localStorage.setItem(STORAGE_KEYS.sessions, JSON.stringify(toStore));
        return;
      } catch (e) {
        // Over quota: drop the oldest session and retry
        if (e instanceof DOMException && toStore.length > 1) {
          toStore = toStore.slice(0, -1);
          continue;
        }
        throw e;
      }
    }
    window.localStorage.removeItem(STORAGE_KEYS.sessions);
  } catch {
    // Storage unavailable (private mode, disabled): history simply is not persisted
  }
}

export function loadSidebarCollapsed(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEYS.sidebarCollapsed) === 'true';
  } catch {
    return false;
  }
}

export function saveSidebarCollapsed(collapsed: boolean): void {
  try {
    window.localStorage.setItem(STORAGE_KEYS.sidebarCollapsed, String(collapsed));
  } catch {
    // ignore: preference just won't persist
  }
}

function newId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
  }
}

export function createSession(now = new Date()): ChatSession {
  const ts = now.toISOString();
  return { id: newId(), title: NEW_CHAT_TITLE, createdAt: ts, updatedAt: ts, messages: [] };
}

export function titleFromPrompt(prompt: string): string {
  const clean = prompt.replace(/\s+/g, ' ').trim();
  if (!clean) return NEW_CHAT_TITLE;
  return clean.length > TITLE_MAX_CHARS ? `${clean.slice(0, TITLE_MAX_CHARS - 1)}…` : clean;
}
