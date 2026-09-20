import type { EvidenceObject } from '../../lib/jobResponse';
import { useT } from '../../i18n/useT';
import type { Translate } from '../../i18n/I18nProvider';

interface EvidenceChipsProps {
  evidence: EvidenceObject[];
  rejectedIds: Set<string>;
  selectedEvidenceId: string | null;
  onSelect: (id: string | null) => void;
}

/** Engines the backend names in source_model, written the way their authors write them. */
const ENGINE_NAMES: Record<string, string> = {
  PALIGEMMA_VQA: 'PaliGemma VQA',
  PALIGEMMA: 'PaliGemma',
  CLASSICAL_CHANGE_DETECTION: 'Classical change detection',
  CHANGEMAMBA: 'ChangeMamba',
  CROMA: 'CROMA',
};

/** Short engine name for a chip: "CLASSICAL_CHANGE_DETECTION (SAR log-ratio …)" → "Classical change detection". */
function engineLabel(t: Translate, sourceModel: string | undefined): string {
  if (!sourceModel) return t('evidence.unknownEngine');
  const key = sourceModel.split('(')[0].trim().replace(/_TOOL$/, '').toUpperCase();
  if (ENGINE_NAMES[key]) return ENGINE_NAMES[key];
  const base = key.replace(/_/g, ' ').toLowerCase();
  return base.charAt(0).toUpperCase() + base.slice(1);
}

function regionLabel(t: Translate, ev: EvidenceObject, index: number): string {
  if (!ev.spatial_region) return t('evidence.noRegion');
  if (ev.processing_parameters?.spatial_region_is_full_image) return t('evidence.wholeImage');
  return t('evidence.regionR', { n: index });
}

/**
 * One chip per evidence object backing the answer (MC5 evidence objects referenced by MC7.1).
 * Selecting a chip highlights its region on the map and opens the matching part of the trace.
 */
export default function EvidenceChips({ evidence, rejectedIds, selectedEvidenceId, onSelect }: EvidenceChipsProps) {
  const t = useT();
  if (evidence.length === 0) return null;
  let regionCount = 0;

  return (
    <div className="chips" aria-label={t('evidence.aria')}>
      {evidence.map((ev) => {
        const localized = !!ev.spatial_region && !ev.processing_parameters?.spatial_region_is_full_image;
        if (localized) regionCount += 1;
        const selected = selectedEvidenceId === ev.evidence_id;
        const rejected = rejectedIds.has(ev.evidence_id);
        return (
          <button
            key={ev.evidence_id}
            type="button"
            className={`chip ${selected ? 'chip--selected' : ''} ${rejected ? 'chip--rejected' : ''}`}
            aria-pressed={selected}
            title={ev.claim}
            onClick={() => onSelect(selected ? null : ev.evidence_id)}
          >
            <span className="chip__engine">{engineLabel(t, ev.source_model)}</span>
            <span className="chip__region">{regionLabel(t, ev, regionCount)}</span>
            {rejected && <span className="chip__rejected">{t('evidence.rejected')}</span>}
          </button>
        );
      })}
    </div>
  );
}
