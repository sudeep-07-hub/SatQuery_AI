interface QueryBoxProps {
  query: string;
  onChange: (query: string) => void;
  error: string | null;
}

export default function QueryBox({ query, onChange, error }: QueryBoxProps) {
  return (
    <div className="query-box">
      <label className="query-box__label" htmlFor="query-input">
        Natural-Language Query
      </label>
      <textarea
        id="query-input"
        className="query-box__textarea"
        placeholder="e.g. Has built-up area increased between these two dates?"
        value={query}
        onChange={(e) => onChange(e.target.value)}
        maxLength={500}
      />
      <div className="query-box__footer">
        <span className="query-box__charcount">{query.length}/500</span>
      </div>
      {error && (
        <div className="error-msg" id="query-error">
          <span className="error-msg__icon">⚠️</span>
          {error}
        </div>
      )}
    </div>
  );
}
