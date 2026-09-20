/**
 * One-click sample queries.
 *
 * Each sample bundles real imagery shipped with the site and a real question. Clicking one uploads
 * those files to the backend and runs the pipeline live — nothing here is a recorded or fabricated
 * answer. `requiresTool` lets the UI disable a sample when the connected backend cannot run it.
 */
import type { TranslationKey } from '../i18n/types';

export interface SampleQuery {
  id: string;
  titleKey: TranslationKey;
  /**
   * The literal string POSTed to /api/query. It stays English in Phase 2: translating it would
   * send a non-English query into a pipeline whose planner and registry rules are English-only,
   * which is the Phase 3 work and is separately gated. The card shows it verbatim, so what the
   * user reads is what the pipeline receives.
   */
  query: string;
  files: string[];
  /** Provenance of the bundled imagery, shown on the card. */
  source: string;
  /** Tool id from GET /api/system that must be available for this sample to run. */
  requiresTool: string;
}

export const SAMPLE_QUERIES: SampleQuery[] = [
  {
    id: 'airstrip',
    titleKey: 'sample.airstrip.title',
    query: 'Has a new airstrip been cleared between these two acquisitions?',
    files: ['/samples/s1aad_ID32_before.tif', '/samples/s1aad_ID32_after.tif'],
    source: 'Sentinel-1 SAR pair · S1-AAD airstrip dataset',
    requiresTool: 'classical_change_detection',
  },
  {
    id: 'change',
    titleKey: 'sample.change.title',
    query: 'What changed between these two images?',
    files: ['/samples/s1aad_ID443_before.tif', '/samples/s1aad_ID443_after.tif'],
    source: 'Sentinel-1 SAR pair · S1-AAD airstrip dataset',
    requiresTool: 'classical_change_detection',
  },
  {
    id: 'river',
    titleKey: 'sample.river.title',
    query: 'Is there a river in this image?',
    files: ['/samples/sentinel2_serbia_26_19.tif'],
    source: 'Sentinel-2 optical patch · BigEarthNet',
    requiresTool: 'single_image_vqa',
  },
];

/** Fetch a sample's bundled files as uploadable File objects. */
export async function loadSampleFiles(sample: SampleQuery): Promise<File[]> {
  return Promise.all(
    sample.files.map(async (path) => {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`sample image missing: ${path}`);
      const blob = await response.blob();
      return new File([blob], path.split('/').pop() as string, { type: 'image/tiff' });
    })
  );
}
