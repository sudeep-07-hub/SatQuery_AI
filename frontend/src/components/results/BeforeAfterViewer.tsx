import { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON, ImageOverlay, useMap } from 'react-leaflet';
import type { LatLngBoundsExpression } from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface ObservationInfo {
  filename: string;
  georeferenced: boolean;
  bounds_wgs84: [[number, number], [number, number]] | null;
  preview_url: string;
}

interface BeforeAfterViewerProps {
  apiBase: string;
  result: any;
  evidenceGraph: any;
  selectedEvidenceId?: string | null;
}

function FitBounds({ bounds }: { bounds: LatLngBoundsExpression }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(bounds, { padding: [16, 16] });
  }, [map, bounds]);
  return null;
}

/** Drag-divider comparison of the two observations (bi-temporal pair, or optical vs SAR). */
function SwipeCompare({ apiBase, left, right, leftLabel, rightLabel }: {
  apiBase: string; left: ObservationInfo; right: ObservationInfo; leftLabel: string; rightLabel: string;
}) {
  const [position, setPosition] = useState(50);
  const frameRef = useRef<HTMLDivElement>(null);

  const setFromClientX = (clientX: number) => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPosition(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  };

  return (
    <div className="swipe">
      <div
        className="swipe__frame"
        ref={frameRef}
        onPointerMove={(e) => { if (e.buttons === 1) setFromClientX(e.clientX); }}
        onPointerDown={(e) => setFromClientX(e.clientX)}
      >
        <img className="swipe__image" src={`${apiBase}${right.preview_url}`} alt={rightLabel} draggable={false} />
        {/* Clipped by paint, not by box size, so both layers keep the identical geometry and stay registered */}
        <div className="swipe__clip" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}>
          <img className="swipe__image" src={`${apiBase}${left.preview_url}`} alt={leftLabel} draggable={false} />
        </div>
        <div className="swipe__divider" style={{ insetInlineStart: `${position}%` }} aria-hidden="true" />
        <span className="swipe__tag swipe__tag--left">{leftLabel}</span>
        <span className="swipe__tag swipe__tag--right">{rightLabel}</span>
      </div>
      <input
        className="swipe__range"
        type="range"
        min={0}
        max={100}
        value={position}
        onChange={(e) => setPosition(Number(e.target.value))}
        aria-label={`Reveal ${leftLabel} over ${rightLabel}`}
      />
    </div>
  );
}

export default function BeforeAfterViewer({ apiBase, result, evidenceGraph, selectedEvidenceId }: BeforeAfterViewerProps) {
  const observations: Record<string, ObservationInfo> = result?.observations || {};
  const obsIds = Object.keys(observations).sort();
  const overlay = result?.change_overlay;

  const roleLabel = (id: string) => {
    if (overlay?.before_observation === id) return 'Before';
    if (overlay?.after_observation === id) return 'After';
    return id.replace('_', ' ');
  };

  const [activeObs, setActiveObs] = useState<string>(overlay?.after_observation || obsIds[0]);
  const [mode, setMode] = useState<'map' | 'swipe'>('map');
  const [showMask, setShowMask] = useState<boolean>(true);
  const [showRegions, setShowRegions] = useState<boolean>(true);

  if (obsIds.length === 0) return null;
  const active = observations[activeObs] || observations[obsIds[0]];

  const features = (evidenceGraph?.nodes || [])
    .filter((n: any) => n.type === 'evidence' && n.data?.spatial_region && !n.data?.processing_parameters?.spatial_region_is_full_image)
    .map((n: any) => ({
      type: 'Feature',
      geometry: n.data.spatial_region,
      properties: { evidence_id: n.data.evidence_id },
    }));

  const featureStyle = (feature: any) => {
    const selected = selectedEvidenceId && feature?.properties?.evidence_id === selectedEvidenceId;
    if (selectedEvidenceId && !selected) return { color: '#9ca3af', weight: 1, fillOpacity: 0 };
    return { color: selected ? '#3b82f6' : '#facc15', weight: selected ? 4 : 2, fillOpacity: selected ? 0.15 : 0 };
  };

  const controls = (
    <div className="viewer-controls">
      {obsIds.length > 1 && (
        <div className="toggle-group toggle-group--mode">
          <button className={`btn-toggle ${mode === 'map' ? 'active' : ''}`} onClick={() => setMode('map')}>Map</button>
          <button className={`btn-toggle ${mode === 'swipe' ? 'active' : ''}`} onClick={() => setMode('swipe')}>Swipe</button>
        </div>
      )}
      {obsIds.length > 1 && (
        <div className="toggle-group">
          {obsIds.map((id) => (
            <button key={id} className={`btn-toggle ${activeObs === id ? 'active' : ''}`} onClick={() => setActiveObs(id)}>
              {roleLabel(id)}
            </button>
          ))}
        </div>
      )}
      {overlay && (
        <label className="viewer-check">
          <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} />
          Change mask
        </label>
      )}
      {features.length > 0 && (
        <label className="viewer-check">
          <input type="checkbox" checked={showRegions} onChange={(e) => setShowRegions(e.target.checked)} />
          Evidence regions
        </label>
      )}
    </div>
  );

  if (mode === 'swipe' && obsIds.length > 1) {
    const leftId = overlay?.before_observation ?? obsIds[0];
    const rightId = overlay?.after_observation ?? obsIds[1];
    return (
      <div className="before-after-viewer">
        {controls}
        <SwipeCompare
          apiBase={apiBase}
          left={observations[leftId]}
          right={observations[rightId]}
          leftLabel={roleLabel(leftId)}
          rightLabel={roleLabel(rightId)}
        />
        <div className="viewer-note">Drag to compare {roleLabel(leftId).toLowerCase()} and {roleLabel(rightId).toLowerCase()}.</div>
      </div>
    );
  }

  if (!active.georeferenced || !active.bounds_wgs84) {
    return (
      <div className="before-after-viewer">
        {controls}
        <div className="plain-viewer">
          <img src={`${apiBase}${active.preview_url}`} alt={active.filename} />
          {overlay && showMask && <img className="plain-viewer__overlay mask-reveal" src={`${apiBase}${overlay.url}`} alt="change mask" />}
        </div>
        <div className="viewer-note">Not georeferenced: shown in image coordinates.</div>
      </div>
    );
  }

  const bounds = active.bounds_wgs84 as LatLngBoundsExpression;
  return (
    <div className="before-after-viewer">
      {controls}
      <div className="map-view">
        <MapContainer bounds={bounds} style={{ height: '100%', width: '100%' }}>
          <FitBounds bounds={bounds} />
          <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <ImageOverlay key={activeObs} url={`${apiBase}${active.preview_url}`} bounds={bounds} opacity={0.95} />
          {overlay && showMask && overlay.bounds_wgs84 && (
            <ImageOverlay
              url={`${apiBase}${overlay.url}`}
              bounds={overlay.bounds_wgs84}
              opacity={0.9}
              className="mask-reveal"
            />
          )}
          {showRegions && features.map((feature: any) => (
            <GeoJSON key={`${feature.properties.evidence_id}-${selectedEvidenceId}`} data={feature} style={() => featureStyle(feature)} />
          ))}
        </MapContainer>
      </div>
      <div className="viewer-note">{active.filename} · {roleLabel(activeObs)}</div>
    </div>
  );
}
