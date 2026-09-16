import type { ChatSession } from '../../lib/chatStorage';

interface ChatThreadProps {
  session: ChatSession | null;
}

// Phase 3: read-only rendering of stored turns. Phase 4 replaces this with the full chat panel.
export default function ChatThread({ session }: ChatThreadProps) {
  if (!session || session.messages.length === 0) {
    return null;
  }

  return (
    <section className="chat-thread" aria-label={`Conversation: ${session.title}`}>
      {session.messages.map((message, idx) => (
        <div key={`${message.timestamp}-${idx}`} className={`chat-turn chat-turn--${message.role}`}>
          <div className="chat-turn__bubble">
            <div className="chat-turn__content">{message.content}</div>
            {message.attachments && message.attachments.length > 0 && (
              <div className="chat-turn__attachments">
                {message.attachments.map((a) => a.name).join(' · ')}
              </div>
            )}
            {message.role === 'assistant' && (message.status || message.confidence) && (
              <div className="chat-turn__meta">
                {message.status && message.status !== 'DONE' && <span>{message.status}</span>}
                {message.confidence && (
                  <span>model confidence (uncalibrated): {(message.confidence.value * 100).toFixed(0)}%</span>
                )}
              </div>
            )}
          </div>
        </div>
      ))}
    </section>
  );
}
