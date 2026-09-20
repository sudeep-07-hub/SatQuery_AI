import { useCallback, useLayoutEffect, useRef } from 'react';
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

/**
 * Attachment chip. PNG/JPEG show the real image; browsers cannot decode GeoTIFF, so those show a
 * labelled tile until the backend returns its own rendering of the raster.
 */
function AttachmentChip({ file, index, count, disabled, onRemove }: {
  file: UploadedFile; index: number; count: number; disabled: boolean; onRemove: () => void;
}) {
  return (
    <span className="attachment">
      <span className="attachment__thumb">
        {file.previewUrl
          ? <img src={file.previewUrl} alt="" />
          : <span className="attachment__thumb-label">TIFF</span>}
      </span>
      <span className="attachment__text">
        <span className="attachment__role">{attachmentLabel(index, count)}</span>
        <span className="attachment__name" title={file.file.name}>{file.file.name}</span>
        <span className="attachment__size">{formatSize(file.file.size)}</span>
      </span>
      <button className="attachment__remove" onClick={onRemove} disabled={disabled} aria-label={`Remove ${file.file.name}`}>×</button>
    </span>
  );
}

export default function ChatComposer({
  files, onFilesChange, prompt, onPromptChange, onSend, inFlight, error, onError,
}: ChatComposerProps) {
  const {
    inputRef, isDragActive, handleDrop, handleDragOver, handleDragLeave, handleInputChange, handleRemove,
    acceptAttribute, maxFiles,
  } = useImageAttachments(files, onFilesChange, onError);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  /**
   * Auto-grow: one control-height at a single line, growing with the content up to
   * --composer-field-max-h, past which CSS caps it and it scrolls.
   */
  const fitToContent = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    // The placeholder counts towards scrollHeight, which is what we want: the field is as tall
    // as whatever it is showing. On a narrow screen the placeholder wraps to two lines and the
    // empty field is two lines tall, exactly as it would be with two lines typed into it.
    el.style.height = 'auto';
    const borders = el.offsetHeight - el.clientHeight; // scrollHeight excludes them
    el.style.height = `${el.scrollHeight + borders}px`;
  }, []);

  useLayoutEffect(fitToContent, [prompt, fitToContent]);

  // The same text wraps to a different number of lines when the field gets narrower, so the
  // height is re-fitted on width changes (resize, rotation, attachment chips) too, not only
  // on input. Width is compared explicitly so re-fitting the height cannot re-trigger this.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    let lastWidth = el.getBoundingClientRect().width;
    const observer = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      if (width === lastWidth) return;
      lastWidth = width;
      fitToContent();
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [fitToContent]);

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

      {isDragActive && (
        <div className="chat-composer__scan" aria-hidden="true"><span>Drop imagery to attach</span></div>
      )}

      {files.length > 0 && (
        <div className={`attachments ${files.length === 2 ? 'attachments--pair' : ''}`}>
          {files.map((f, i) => (
            <AttachmentChip
              key={f.id}
              file={f}
              index={i}
              count={files.length}
              disabled={inFlight}
              onRemove={() => handleRemove(f.id)}
            />
          ))}
          {files.length === 2 && <span className="attachments__note">sent together as one bi-temporal pair</span>}
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
        <div className="chat-composer__field">
          <textarea
            ref={textareaRef}
            id="chat-prompt"
            className="chat-composer__input"
            placeholder={files.length === 0 ? 'Attach an image, then ask a question…' : 'Ask a question about the attached image(s)…'}
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            maxLength={500}
          />
          {/* The in-field icon row renders only when it holds an icon; the count it carries is
              what reserves the textarea's trailing padding. The mic joins it in Phase 4. */}
        </div>
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
