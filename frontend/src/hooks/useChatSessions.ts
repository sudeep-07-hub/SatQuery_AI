import { useCallback, useEffect, useState } from 'react';
import {
  type ChatMessage,
  type ChatSession,
  createSession,
  loadSessions,
  loadSidebarCollapsed,
  saveSessions,
  saveSidebarCollapsed,
  sortByRecent,
  titleFromPrompt,
  NEW_CHAT_TITLE,
} from '../lib/chatStorage';

/** Chat sessions + sidebar preference, persisted to localStorage only (no backend calls). */
export function useChatSessions() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => loadSessions());
  const [activeId, setActiveId] = useState<string | null>(() => loadSessions()[0]?.id ?? null);
  const [collapsed, setCollapsed] = useState<boolean>(() => loadSidebarCollapsed());

  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    saveSidebarCollapsed(collapsed);
  }, [collapsed]);

  const activeSession = sessions.find((s) => s.id === activeId) ?? null;

  /** Opens an empty session: reuses an existing empty one instead of stacking blanks. Never clears others. */
  const newChat = useCallback((): string => {
    const existingEmpty = sessions.find((s) => s.messages.length === 0);
    if (existingEmpty) {
      setActiveId(existingEmpty.id);
      return existingEmpty.id;
    }
    const session = createSession();
    setSessions((prev) => sortByRecent([session, ...prev]));
    setActiveId(session.id);
    return session.id;
  }, [sessions]);

  const selectSession = useCallback((id: string) => setActiveId(id), []);

  const deleteSession = useCallback((id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    setActiveId((current) => {
      if (current !== id) return current;
      return sessions.find((s) => s.id !== id)?.id ?? null;
    });
  }, [sessions]);

  /** Appends a message; the first user message becomes the session title. */
  const appendMessage = useCallback((sessionId: string, message: ChatMessage) => {
    setSessions((prev) => sortByRecent(prev.map((s) => {
      if (s.id !== sessionId) return s;
      const title = s.title === NEW_CHAT_TITLE && message.role === 'user' ? titleFromPrompt(message.content) : s.title;
      return { ...s, title, updatedAt: message.timestamp, messages: [...s.messages, message] };
    })));
  }, []);

  /** Ensures there is an active session to write into, creating one if needed. */
  const ensureActiveSession = useCallback((): string => {
    if (activeSession) return activeSession.id;
    const session = createSession();
    setSessions((prev) => sortByRecent([session, ...prev]));
    setActiveId(session.id);
    return session.id;
  }, [activeSession]);

  return {
    sessions,
    activeSession,
    activeId,
    collapsed,
    setCollapsed,
    newChat,
    selectSession,
    deleteSession,
    appendMessage,
    ensureActiveSession,
  };
}
