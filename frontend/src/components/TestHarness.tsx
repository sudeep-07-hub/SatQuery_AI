import type { UploadedFile } from './UploadZone';

interface TestCase {
  label: string;
  id: string;
  files: { url: string; name: string }[];
  query: string;
}

const TEST_CASES: TestCase[] = [
  {
    label: 'A: Single Optical GeoTIFF',
    id: 'test-case-a',
    files: [{ url: '/test_geo1.tif', name: 'sentinel2_optical.tif' }],
    query: 'Detect urban sprawl in this region',
  },
  {
    label: 'B: Two Optical GeoTIFFs',
    id: 'test-case-b',
    files: [
      { url: '/test_geo1.tif', name: 'sentinel2_band1.tif' },
      { url: '/test_geo_optical2.tif', name: 'sentinel2_band2.tif' },
    ],
    query: 'Compare NDVI change between two dates',
  },
  {
    label: 'C: Optical + SAR',
    id: 'test-case-c',
    files: [
      { url: '/test_geo1.tif', name: 'sentinel2_optical.tif' },
      { url: '/test_geo2.tif', name: 'risat_sar.tif' },
    ],
    query: 'Has built-up area increased?',
  },
  {
    label: 'D: Unsupported File',
    id: 'test-case-d',
    files: [{ url: '/test_invalid.pdf', name: 'report.pdf' }],
    query: 'Analyze this document',
  },
  {
    label: 'E: No Georef (PNG)',
    id: 'test-case-e',
    files: [{ url: '/test_plain.png', name: 'aerial_photo.png' }],
    query: 'Identify water bodies',
  },
];

interface TestHarnessProps {
  onLoadCase: (files: UploadedFile[], query: string) => void;
}

export default function TestHarness({ onLoadCase }: TestHarnessProps) {
  const handleClick = async (tc: TestCase) => {
    const uploadedFiles: UploadedFile[] = [];
    for (const f of tc.files) {
      const resp = await fetch(f.url);
      const blob = await resp.blob();
      const file = new File([blob], f.name, { type: blob.type || 'application/octet-stream' });
      const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
      const isTiff = ext === 'tif' || ext === 'tiff';
      uploadedFiles.push({
        file,
        id: `${f.name}-${Date.now()}-${Math.random()}`,
        previewUrl: isTiff ? null : URL.createObjectURL(file),
        isGeoTiff: isTiff,
      });
    }
    onLoadCase(uploadedFiles, tc.query);
  };

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 12,
        left: 12,
        zIndex: 9999,
        background: 'rgba(0,0,0,0.85)',
        border: '1px solid rgba(120,200,255,0.3)',
        borderRadius: 8,
        padding: '8px 12px',
        display: 'flex',
        gap: 6,
        flexWrap: 'wrap',
        maxWidth: 600,
      }}
    >
      <span style={{ color: '#8bb', fontSize: 11, fontWeight: 600, width: '100%' }}>
        🧪 Test Harness
      </span>
      {TEST_CASES.map((tc) => (
        <button
          key={tc.id}
          id={tc.id}
          onClick={() => handleClick(tc)}
          style={{
            fontSize: 11,
            padding: '4px 10px',
            borderRadius: 4,
            border: '1px solid rgba(120,200,255,0.4)',
            background: 'rgba(30,60,90,0.8)',
            color: '#cde',
            cursor: 'pointer',
          }}
        >
          {tc.label}
        </button>
      ))}
    </div>
  );
}
