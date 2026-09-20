import type { ChatSession } from '../../lib/chatStorage';
import { useT } from '../../i18n/useT';

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
  const t = useT();
  // On narrow screens the same control dismisses the drawer instead of shrinking it to a rail.
  const isDrawer = () => window.matchMedia('(max-width: 900px)').matches;

  return (
    <aside
      className={`chat-sidebar ${collapsed ? 'chat-sidebar--collapsed' : ''} ${drawerOpen ? 'chat-sidebar--open' : ''}`}
      aria-label={t('sidebar.aria')}
    >
      <div className="chat-sidebar__header">
        {/* The SatQuery mark stays visible in both states */}
        <span className="chat-sidebar__mark app-header__icon" aria-hidden="true">🛰</span>
        <button
          className="chat-sidebar__toggle"
          onClick={() => (isDrawer() ? onCloseDrawer() : onToggleCollapsed())}
          aria-expanded={!collapsed}
          aria-label={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
          title={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
        >
          <span className={`json-viewer__toggle chat-sidebar__toggle-icon ${collapsed ? '' : 'json-viewer__toggle--open'}`}>▾</span>
        </button>
      </div>

      <button
        className="chat-sidebar__new"
        onClick={onNewChat}
        disabled={busy}
        title={busy ? t('sidebar.newChatBusy') : t('sidebar.newChatTitle')}
        aria-label={t('sidebar.newChat')}
      >
        <span aria-hidden="true">+</span>
        {(!collapsed || drawerOpen) && <span>{t('sidebar.newChat')}</span>}
      </button>

      {(!collapsed || drawerOpen) && (
        <nav className="chat-sidebar__list">
          {sessions.length === 0 ? (
            <div className="chat-sidebar__empty">{t('sidebar.empty')}</div>
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
                  aria-label={t('sidebar.deleteChatAria', { title: session.title })}
                  title={t('sidebar.deleteChat')}
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
