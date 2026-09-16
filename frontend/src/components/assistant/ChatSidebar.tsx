import type { ChatSession } from '../../lib/chatStorage';

interface ChatSidebarProps {
  sessions: ChatSession[];
  activeId: string | null;
  collapsed: boolean;
  /** True while a query is running: switching sessions would orphan the in-flight response. */
  busy: boolean;
  onToggleCollapsed: () => void;
  onNewChat: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function ChatSidebar({
  sessions, activeId, collapsed, busy, onToggleCollapsed, onNewChat, onSelect, onDelete,
}: ChatSidebarProps) {
  return (
    <aside className={`chat-sidebar ${collapsed ? 'chat-sidebar--collapsed' : ''}`} aria-label="Chat history">
      <div className="chat-sidebar__header">
        {/* Same toggle pattern as the JSON viewer (▾ glyph rotated by the --open modifier) */}
        <button
          className="chat-sidebar__toggle"
          onClick={onToggleCollapsed}
          aria-expanded={!collapsed}
          aria-label={collapsed ? 'Expand chat history' : 'Collapse chat history'}
          title={collapsed ? 'Expand chat history' : 'Collapse chat history'}
        >
          <span className={`json-viewer__toggle chat-sidebar__toggle-icon ${collapsed ? '' : 'json-viewer__toggle--open'}`}>▾</span>
        </button>
        <button
          className="chat-sidebar__new"
          onClick={onNewChat}
          disabled={busy}
          title={busy ? 'Wait for the current response to finish' : 'Start a new chat'}
          aria-label="New chat"
        >
          <span aria-hidden="true">+</span>
          {!collapsed && <span>New Chat</span>}
        </button>
      </div>

      {!collapsed && (
        <nav className="chat-sidebar__list">
          {sessions.length === 0 ? (
            <div className="chat-sidebar__empty">No chats yet.</div>
          ) : (
            sessions.map((session) => (
              <div
                key={session.id}
                className={`chat-sidebar__item ${session.id === activeId ? 'chat-sidebar__item--active' : ''}`}
              >
                <button
                  className="chat-sidebar__item-title"
                  onClick={() => onSelect(session.id)}
                  disabled={busy && session.id !== activeId}
                  title={session.title}
                  aria-current={session.id === activeId ? 'true' : undefined}
                >
                  {session.title}
                </button>
                <button
                  className="chat-sidebar__item-delete"
                  onClick={() => onDelete(session.id)}
                  disabled={busy && session.id === activeId}
                  aria-label={`Delete chat "${session.title}"`}
                  title="Delete chat"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </nav>
      )}
    </aside>
  );
}
