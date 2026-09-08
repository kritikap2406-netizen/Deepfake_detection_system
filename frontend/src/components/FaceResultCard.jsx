import StreamScoreBar from './StreamScoreBar'

const STREAM_LABELS = {
  spatial_score: 'Spatial',
  frequency_score: 'Frequency',
  noise_score: 'Noise',
  blink_score: 'Blink',
  rppg_score: 'rPPG',
  temporal_score: 'Temporal',
}

export default function FaceResultCard({ face }) {
  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <h3 style={styles.faceTitle}>Face {face.face_id}</h3>
      </div>

      <div style={styles.body}>
        <img
          src={`/static/heatmaps/${face.heatmap}`}
          alt={`Grad-CAM++ heatmap for face ${face.face_id}`}
          style={styles.heatmap}
        />

        <div style={styles.scores}>
          {Object.entries(STREAM_LABELS).map(([key, label]) => (
            <StreamScoreBar key={key} label={label} value={face[key]} />
          ))}
        </div>
      </div>

      {face.suspicious_regions.length > 0 && (
        <div style={styles.regions}>
          <strong>Suspicious regions: </strong>
          {face.suspicious_regions.map((r) => r.replaceAll('_', ' ')).join(', ')}
        </div>
      )}
    </div>
  )
}

const styles = {
  card: { border: '1px solid #e5e5e5', borderRadius: 10, padding: 16, marginBottom: 16, background: '#fff' },
  header: { marginBottom: 12 },
  faceTitle: { margin: 0, fontSize: '1.05rem' },
  body: { display: 'flex', gap: 20, flexWrap: 'wrap' },
  heatmap: { width: 160, height: 160, borderRadius: 8, objectFit: 'cover', flexShrink: 0, background: '#f0f0f0' },
  scores: { flex: 1, minWidth: 220 },
  regions: { marginTop: 12, fontSize: '0.85rem', color: '#a83232' },
}
