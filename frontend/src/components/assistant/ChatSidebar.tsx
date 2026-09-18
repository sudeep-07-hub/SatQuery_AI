import type { ChatSession } from '../../lib/chatStorage';

interface ChatSidebarProps {
  sessions: ChatSession[];
  activeId: string | null;
  collapsed: boolean;
  /** Narrow screens show the sidebar as an off-canvas drawer; this is its open state. */
  drawerOpen: boolean;
  /** True while a query is running: switching sessions would orphan the in-flight response. */
  busy: boolean;
  onToggleCollapsed: () => void;
  onCloseDrawer: () => void;
  onNewChat: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function ChatSidebar({
  sessions, activeId, collapsed, drawerOpen, busy, onToggleCollapsed, onCloseDrawer, onNewChat, onSelect, onDelete,
}: ChatSidebarProps) {
  // On narrow screens the same control dismisses the drawer instead of shrinking it to a rail.
  const isDrawer = () => window.matchMedia('(max-width: 900px)').matches;

  return (
    <aside
      className={`chat-sidebar ${collapsed ? 'chat-sidebar--collapsed' : ''} ${drawerOpen ? 'chat-sidebar--open' : ''}`}
      aria-label="Chat history"
    >
      <div className="chat-sidebar__header">
        {/* The SatQuery mark stays visible in both states */}
        <span className="chat-sidebar__mark app-header__icon" aria-hidden="true">🛰</span>
        <button
          className="chat-sidebar__toggle"
          onClick={() => (isDrawer() ? onCloseDrawer() : onToggleCollapsed())}
          aria-expanded={!collapsed}
          aria-label={collapsed ? 'Expand chat history' : 'Collapse chat history'}
          title={collapsed ? 'Expand chat history' : 'Collapse chat history'}
        >
          <span className={`json-viewer__toggle chat-sidebar__toggle-icon ${collapsed ? '' : 'json-viewer__toggle--open'}`}>▾</span>
        </button>
      </div>

      <button
        className="chat-sidebar__new"
        onClick={onNewChat}
        disabled={busy}
        title={busy ? 'Wait for the current response to finish' : 'Start a new chat'}
        aria-label="New chat"
      >
        <span aria-hidden="true">+</span>
        {(!collapsed || drawerOpen) && <span>New Chat</span>}
      </button>

      {(!collapsed || drawerOpen) && (
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
