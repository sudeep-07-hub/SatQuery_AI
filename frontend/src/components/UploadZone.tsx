import { useCallback, useRef, useState } from 'react';

interface UploadedFile {
  file: File;
  id: string;
  previewUrl: string | null;
  isGeoTiff: boolean;
}

interface UploadZoneProps {
  files: UploadedFile[];
  onFilesChange: (files: UploadedFile[]) => void;
  error: string | null;
  onError: (error: string | null) => void;
}

const ACCEPTED_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
const MAX_FILES = 2;

function getExtension(name: string): string {
  const idx = name.lastIndexOf('.');
  return idx >= 0 ? name.slice(idx).toLowerCase() : '';
}

function isGeoTiffExtension(ext: string): boolean {
  return ext === '.tif' || ext === '.tiff';
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const METADATA_FIELDS = [
  { key: 'modality', label: 'Modality' },
  { key: 'sensor', label: 'Sensor' },
  { key: 'gsd', label: 'GSD' },
  { key: 'crs', label: 'CRS' },
  { key: 'date', label: 'Acq. Date' },
  { key: 'quality', label: 'Quality' },
];

export type { UploadedFile };

export default function UploadZone({ files, onFilesChange, error, onError }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragActive, setIsDragActive] = useState(false);

  const processFiles = useCallback(
    (incoming: FileList | File[]) => {
      const newFiles: UploadedFile[] = [];

      for (const file of Array.from(incoming)) {
        const ext = getExtension(file.name);
        if (!ACCEPTED_EXTENSIONS.includes(ext)) {
          onError(`Unsupported format: ${ext}. Accepted: .tif, .tiff, .png, .jpg, .jpeg`);
          return;
        }
      }

      if (files.length + incoming.length > MAX_FILES) {
        onError('Maximum 2 images allowed. Remove one before adding another.');
        return;
      }

      onError(null);

      for (const file of Array.from(incoming)) {
        const ext = getExtension(file.name);
        const isTiff = isGeoTiffExtension(ext);
        const previewUrl = isTiff ? null : URL.createObjectURL(file);
        newFiles.push({
          file,
          id: `${file.name}-${Date.now()}-${Math.random()}`,
          previewUrl,
          isGeoTiff: isTiff,
        });
      }

      onFilesChange([...files, ...newFiles]);
    },
    [files, onFilesChange, onError]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragActive(false);
      if (e.dataTransfer.files.length > 0) {
        processFiles(e.dataTransfer.files);
      }
    },
    [processFiles]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        processFiles(e.target.files);
      }
      // Reset input so re-selecting the same file works
      e.target.value = '';
    },
    [processFiles]
  );

  const handleRemove = useCallback(
    (id: string) => {
      const toRemove = files.find((f) => f.id === id);
      if (toRemove?.previewUrl) URL.revokeObjectURL(toRemove.previewUrl);
      onFilesChange(files.filter((f) => f.id !== id));
      onError(null);
    },
    [files, onFilesChange, onError]
  );

  const hasFiles = files.length > 0;

  return (
    <div>
      <div
        className={`upload-zone ${isDragActive ? 'upload-zone--active' : ''} ${hasFiles ? 'upload-zone--has-files' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !hasFiles && inputRef.current?.click()}
        id="upload-zone"
      >
        <input
          ref={inputRef}
          type="file"
          className="upload-zone__input"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          multiple
          onChange={handleInputChange}
          id="file-input"
        />

        {!hasFiles ? (
          <>
            <div className="upload-zone__icon">🛰️</div>
            <div className="upload-zone__text">
              Drop remote-sensing images here, or click to browse
            </div>
            <div className="upload-zone__hint">
              .tif · .tiff · .png · .jpg · .jpeg — up to 2 images
            </div>
          </>
        ) : (
          <>
            <div className={`file-cards ${files.length === 1 ? 'file-cards--single' : ''}`}>
              {files.map((f, idx) => (
                <div className="file-card" key={f.id}>
                  <button
                    className="file-card__remove"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemove(f.id);
                    }}
                    title="Remove file"
                    id={`remove-file-${idx}`}
                  >
                    ×
                  </button>

                  {f.previewUrl ? (
                    <div className="file-card__preview">
                      <img src={f.previewUrl} alt={f.file.name} />
                    </div>
                  ) : (
                    <div className="file-card__fallback">
                      <div className="file-card__fallback-icon">🌍</div>
                      <div className="file-card__fallback-label">GeoTIFF</div>
                    </div>
                  )}

                  <div className="file-card__info">
                    <div className="file-card__name" title={f.file.name}>
                      {f.file.name}
                    </div>
                    <div className="file-card__size">{formatSize(f.file.size)}</div>
                  </div>

                  <div className="file-card__meta">
                    {METADATA_FIELDS.map((m) => (
                      <div className="meta-item" key={m.key}>
                        <span className="meta-item__label">{m.label}</span>
                        <span className="meta-item__value">—</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {files.length < MAX_FILES && (
              <div className="upload-zone__add-more">
                <button
                  className="upload-zone__add-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    inputRef.current?.click();
                  }}
                  id="add-more-btn"
                >
                  + Add second image
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {error && (
        <div className="error-msg" id="upload-error">
          <span className="error-msg__icon">⚠️</span>
          {error}
        </div>
      )}
    </div>
  );
}
