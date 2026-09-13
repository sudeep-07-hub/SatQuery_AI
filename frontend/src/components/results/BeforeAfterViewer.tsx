import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, ImageOverlay } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

interface BeforeAfterViewerProps {
  evidenceGraph: any;
  files: any[];
  jobId: string;
  selectedEvidenceId?: string | null;
}

export default function BeforeAfterViewer({ evidenceGraph, files, jobId, selectedEvidenceId }: BeforeAfterViewerProps) {
  const [activeImage, setActiveImage] = useState<number>(0);
  const [showMask, setShowMask] = useState<boolean>(true);
  const [showRegions, setShowRegions] = useState<boolean>(true);

  // Extract GeoJSON features from Evidence Graph, attaching evidence_id for styling
  const features: any[] = [];
  if (evidenceGraph && evidenceGraph.nodes) {
    for (const node of evidenceGraph.nodes) {
      if (node.type === 'evidence' && node.data?.spatial_region) {
        // Clone feature and inject evidence_id into properties
        const feature = {
          ...node.data.spatial_region,
          properties: {
            ...node.data.spatial_region.properties,
            evidence_id: node.data.evidence_id
          }
        };
        features.push(feature);
      }
    }
  }

  // Calculate bounding box of all features to set map bounds
  let center: [number, number] = [0, 0];
  let zoom = 2;
  let bounds: [[number, number], [number, number]] | null = null;
  
  if (features.length > 0 && features[0].geometry && features[0].geometry.coordinates) {
    const coords = features[0].geometry.coordinates[0];
    if (coords && coords.length > 0) {
      center = [coords[0][1], coords[0][0]]; // Leaflet uses [lat, lng]
      zoom = 14;
      
      // Calculate bounds for ImageOverlay (min/max lat/lng)
      let minLat = 90, maxLat = -90, minLng = 180, maxLng = -180;
      for (const pt of coords) {
        if (pt[1] < minLat) minLat = pt[1];
        if (pt[1] > maxLat) maxLat = pt[1];
        if (pt[0] < minLng) minLng = pt[0];
        if (pt[0] > maxLng) maxLng = pt[0];
      }
      bounds = [[minLat, minLng], [maxLat, maxLng]];
    }
  }

  const getFeatureStyle = (feature: any) => {
    const featureId = feature.properties?.evidence_id;
    if (selectedEvidenceId) {
      if (featureId === selectedEvidenceId) {
        return { color: '#3b82f6', weight: 4, fillColor: '#3b82f6', fillOpacity: 0.2 };
      }
      return { color: '#9ca3af', weight: 1, fillColor: 'transparent' };
    }
    return { color: '#ef4444', weight: 2, fillColor: 'transparent' };
  };

  return (
    <div className="before-after-viewer">
      <div className="viewer-controls" style={{ display: 'flex', gap: '15px', marginBottom: '10px', flexWrap: 'wrap' }}>
        {files.length > 1 && (
          <div className="toggle-group" style={{ display: 'flex', gap: '5px' }}>
            <button 
              className={`btn-toggle ${activeImage === 0 ? 'active' : ''}`}
              onClick={() => setActiveImage(0)}
              style={{ padding: '4px 12px', borderRadius: '4px', border: '1px solid #ccc', background: activeImage === 0 ? '#3b82f6' : '#fff', color: activeImage === 0 ? '#fff' : '#333', cursor: 'pointer' }}
            >
              Time 1
            </button>
            <button 
              className={`btn-toggle ${activeImage === 1 ? 'active' : ''}`}
              onClick={() => setActiveImage(1)}
              style={{ padding: '4px 12px', borderRadius: '4px', border: '1px solid #ccc', background: activeImage === 1 ? '#3b82f6' : '#fff', color: activeImage === 1 ? '#fff' : '#333', cursor: 'pointer' }}
            >
              Time 2
            </button>
          </div>
        )}
        
        <div className="toggle-group" style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '14px', cursor: 'pointer' }}>
            <input 
              type="checkbox" 
              checked={showMask} 
              onChange={e => setShowMask(e.target.checked)} 
            />
            Show Change Mask
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '14px', cursor: 'pointer' }}>
            <input 
              type="checkbox" 
              checked={showRegions} 
              onChange={e => setShowRegions(e.target.checked)} 
            />
            Show Evidence Regions
          </label>
        </div>
      </div>

      <div className="map-view" style={{ height: '400px', width: '100%', borderRadius: '8px', overflow: 'hidden', position: 'relative', border: '1px solid #e5e7eb' }}>
        <MapContainer center={center} zoom={zoom} style={{ height: '100%', width: '100%' }}>
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {showRegions && features.map((feature, idx) => (
            <GeoJSON 
              key={`region-${idx}-${selectedEvidenceId}`} 
              data={feature} 
              style={() => getFeatureStyle(feature)}
            />
          ))}

          {showMask && bounds && (
            <ImageOverlay
              url={`http://localhost:8000/api/jobs/${jobId}/export/png`}
              bounds={bounds}
              opacity={0.6}
            />
          )}
        </MapContainer>
      </div>
    </div>
  );
}
