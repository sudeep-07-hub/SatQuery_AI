interface AnalyzeButtonProps {
  disabled: boolean;
  loading: boolean;
  onClick: () => void;
}

export default function AnalyzeButton({ disabled, loading, onClick }: AnalyzeButtonProps) {
  return (
    <button
      className="analyze-btn"
      disabled={disabled || loading}
      onClick={onClick}
      id="analyze-btn"
    >
      {loading ? (
        <>
          <div className="analyze-btn__spinner" />
          Analyzing…
        </>
      ) : (
        <>
          <span>◉</span>
          Analyze
        </>
      )}
    </button>
  );
}
