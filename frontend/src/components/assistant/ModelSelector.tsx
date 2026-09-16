// Placeholder selector: exactly one real controller today. The specialist engines behind it
// (VQA, change detection, fusion) are internal pipeline components chosen by the tool registry,
// not user-selectable models, so they are deliberately not listed here.
export const ACTIVE_CONTROLLER = {
  value: 'satquery-agentic-controller-v1',
  label: 'SatQuery Agentic Controller — v1',
};

export default function ModelSelector() {
  return (
    <label className="model-selector">
      <span className="model-selector__label">Controller</span>
      <select
        className="model-selector__select"
        defaultValue={ACTIVE_CONTROLLER.value}
        aria-label="Controller"
        title="Only one controller exists today; the selection does not change any request yet."
      >
        <option value={ACTIVE_CONTROLLER.value}>{ACTIVE_CONTROLLER.label}</option>
        <option value="coming-soon" disabled>More controllers — coming soon (not available)</option>
      </select>
    </label>
  );
}
