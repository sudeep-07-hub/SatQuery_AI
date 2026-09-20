import { useT } from '../../i18n/useT';

// Placeholder selector: exactly one real controller today. The specialist engines behind it
// (VQA, change detection, fusion) are internal pipeline components chosen by the tool registry,
// not user-selectable models, so they are deliberately not listed here.
export const ACTIVE_CONTROLLER = {
  value: 'satquery-agentic-controller-v1',
  label: 'SatQuery Agentic Controller — v1',
};

export default function ModelSelector() {
  const t = useT();
  return (
    <label className="model-selector">
      <span className="model-selector__label">{t('controller.label')}</span>
      <select
        className="model-selector__select"
        defaultValue={ACTIVE_CONTROLLER.value}
        aria-label={t('controller.label')}
        title={t('controller.onlyOne')}
      >
        <option value={ACTIVE_CONTROLLER.value}>{ACTIVE_CONTROLLER.label}</option>
        <option value="coming-soon" disabled>{t('controller.comingSoon')}</option>
      </select>
    </label>
  );
}
