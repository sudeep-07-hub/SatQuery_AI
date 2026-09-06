import { useState } from 'react';

interface ProfilePanelProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  profile: Record<string, any> | null;
}

function getBarClass(value: number): string {
  if (value >= 0.85) return 'metric__bar-fill--high';
  if (value >= 0.6) return 'metric__bar-fill--medium';
  return 'metric__bar-fill--low';
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(0)}%`;
}

function formatScore(value: number | null | undefined): string {
  if (value == null) return '—';
  return value.toFixed(2);
}

/** Syntax-highlight a JSON string */
function highlightJson(json: string): string {
  return json.replace(
    /("(?:[^"\\]|\\.)*")\s*:/g,
    '<span class="json-key">$1</span>:'
  ).replace(
    /:\s*("(?:[^"\\]|\\.)*")/g,
    (match, val) => `: <span class="json-string">${val}</span>`
  ).replace(
    /:\s*(\d+\.?\d*)/g,
    ': <span class="json-number">$1</span>'
  ).replace(
    /:\s*(true|false)/g,
    ': <span class="json-boolean">$1</span>'
  ).replace(
    /:\s*(null)/g,
    ': <span class="json-null">$1</span>'
  );
}

function RelationshipBadge({ relationship }: { relationship: string }) {
  const label = relationship.replace(/_/g, ' ');
  const badgeClass =
    relationship === 'cross_modal'
      ? 'summary-card__badge--cross-modal'
      : 'summary-card__badge--single';

  return (
    <span className={`summary-card__badge ${badgeClass}`}>
      {relationship === 'cross_modal' ? '◈' : '◇'} {label}
    </span>
  );
}

export default function ProfilePanel({ profile }: ProfilePanelProps) {
  const [jsonExpanded, setJsonExpanded] = useState(true);

  if (!profile) {
    return (
      <div className="profile-empty" id="profile-empty">
        <div className="profile-empty__icon">📡</div>
        <div className="profile-empty__text">
          Upload images and enter a query, then click <strong>Analyze</strong> to generate a
          Structured Input Profile.
        </div>
      </div>
    );
  }

  const jsonString = JSON.stringify(profile, null, 2);
  const highlighted = highlightJson(jsonString);
  const quality = profile.quality as Record<string, number> | undefined;

  return (
    <div id="profile-result">
      {/* Summary Card */}
      <div className="summary-card" id="summary-card">
        <div className="summary-card__header">
          <RelationshipBadge relationship={profile.relationship} />
          <span className="summary-card__count">
            {profile.image_count} image{profile.image_count > 1 ? 's' : ''}
          </span>
        </div>

        <div className="summary-card__metrics">
          {/* Spatial Overlap */}
          <div className="metric">
            <span className="metric__label">Spatial Overlap</span>
            <span className="metric__value">{formatPercent(profile.spatial_overlap)}</span>
            {profile.spatial_overlap != null && (
              <div className="metric__bar">
                <div
                  className={`metric__bar-fill ${getBarClass(profile.spatial_overlap)}`}
                  style={{ width: `${profile.spatial_overlap * 100}%` }}
                />
              </div>
            )}
          </div>

          {/* Co-registration Score */}
          <div className="metric">
            <span className="metric__label">Co-registration</span>
            <span className="metric__value">{formatScore(profile.coregistration_score)}</span>
            {profile.coregistration_score != null && (
              <div className="metric__bar">
                <div
                  className={`metric__bar-fill ${getBarClass(profile.coregistration_score)}`}
                  style={{ width: `${profile.coregistration_score * 100}%` }}
                />
              </div>
            )}
          </div>

          {/* Quality Scores */}
          <div className="metric" style={{ gridColumn: '1 / -1' }}>
            <span className="metric__label">Quality Scores</span>
            {quality && (
              <div className="quality-grid">
                {Object.entries(quality).map(([modality, score]) => (
                  <div className="quality-badge" key={modality}>
                    <span className="quality-badge__label">{modality}</span>
                    <span className="quality-badge__score">{formatScore(score)}</span>
                    <div className="metric__bar" style={{ width: 60 }}>
                      <div
                        className={`metric__bar-fill ${getBarClass(score)}`}
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* JSON Viewer */}
      <div className="json-viewer" id="json-viewer">
        <div
          className="json-viewer__header"
          onClick={() => setJsonExpanded((v) => !v)}
        >
          <span className="json-viewer__title">
            <span>{ }</span> Raw JSON
          </span>
          <span
            className={`json-viewer__toggle ${jsonExpanded ? 'json-viewer__toggle--open' : ''}`}
          >
            ▾
          </span>
        </div>
        {jsonExpanded && (
          <div className="json-viewer__body">
            <pre
              className="json-viewer__pre"
              dangerouslySetInnerHTML={{ __html: highlighted }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
