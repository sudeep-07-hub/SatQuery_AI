import type { KeyboardEvent } from 'react';
import { formatSize, useImageAttachments, type UploadedFile } from '../UploadZone';
import { attachmentLabel } from './ChatThread';

interface ChatComposerProps {
  files: UploadedFile[];
  onFilesChange: (files: UploadedFile[]) => void;
  prompt: string;
  onPromptChange: (value: string) => void;
  onSend: () => void;
  inFlight: boolean;
  error: string | null;
  onError: (error: string | null) => void;
}

export default function ChatComposer({
  files, onFilesChange, prompt, onPromptChange, onSend, inFlight, error, onError,
}: ChatComposerProps) {
  const {
    inputRef, isDragActive, handleDrop, handleDragOver, handleDragLeave, handleInputChange, handleRemove,
    acceptAttribute, maxFiles,
  } = useImageAttachments(files, onFilesChange, onError);

  const canSend = !inFlight && files.length > 0 && prompt.trim().length > 0;

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (canSend) onSend();
    }
  };

  return (
    <div
      className={`chat-composer ${isDragActive ? 'chat-composer--drag' : ''}`}
      onDrop={inFlight ? undefined : handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <input
        ref={inputRef}
        id="chat-file-input"
        type="file"
        className="upload-zone__input"
        accept={acceptAttribute}
        multiple
        onChange={handleInputChange}
      />

      {files.length > 0 && (
        <div className="chat-chips chat-composer__chips">
          {files.map((f, i) => (
            <span key={f.id} className="chat-chip">
              <span className="chat-chip__label">{attachmentLabel(i, files.length)}</span>
              <span className="chat-chip__name" title={f.file.name}>{f.file.name}</span>
              <span className="chat-chip__size">{formatSize(f.file.size)}</span>
              <button
                className="chat-chip__remove"
                onClick={() => handleRemove(f.id)}
                disabled={inFlight}
                aria-label={`Remove ${f.file.name}`}
              >
                ×
              </button>
            </span>
          ))}
          {files.length === 2 && <span className="chat-composer__pair-note">Both images are sent together.</span>}
        </div>
      )}

      <div className="chat-composer__row">
        <button
          className="chat-composer__attach"
          onClick={() => inputRef.current?.click()}
          disabled={inFlight || files.length >= maxFiles}
          aria-label="Attach images"
          title={files.length >= maxFiles ? 'Maximum 2 images' : 'Attach GeoTIFF, PNG or JPEG (or drop files here)'}
        >
          +
        </button>
        <textarea
          id="chat-prompt"
          className="chat-composer__input"
          placeholder={files.length === 0 ? 'Attach an image, then ask a question…' : 'Ask a question about the attached image(s)…'}
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          maxLength={500}
        />
        <button className="chat-composer__send" onClick={onSend} disabled={!canSend}>
          {inFlight ? (<><span className="chat-spinner" aria-hidden="true" /> Analyzing…</>) : 'Send'}
        </button>
      </div>

      {error && (
        <div className="error-msg" role="alert">
          <span className="error-msg__icon">⚠️</span>
          {error}
        </div>
      )}
    </div>
  );
}
