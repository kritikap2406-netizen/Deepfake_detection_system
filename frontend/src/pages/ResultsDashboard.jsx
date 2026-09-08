import FaceResultCard from '../components/FaceResultCard'

export default function ResultsDashboard({ result, onReset }) {
  if (result.prediction === 'UNKNOWN') {
    return (
      <div style={styles.container}>
        <p style={styles.noFace}>{result.message}</p>
        <button style={styles.resetButton} onClick={onReset}>Analyze another file</button>
      </div>
    )
  }

  const isFake = result.prediction === 'FAKE'

  return (
    <div style={styles.container}>
      <div style={{ ...styles.verdictCard, borderColor: isFake ? '#e05a4e' : '#3aa96b' }}>
        <div>
          <span style={{ ...styles.verdictLabel, color: isFake ? '#e05a4e' : '#3aa96b' }}>
            {result.prediction}
          </span>
          <p style={styles.confidence}>Calibrated confidence: {(result.confidence * 100).toFixed(1)}%</p>
        </div>
      </div>

      <div style={styles.explanationBox}>
        <h4 style={styles.sectionTitle}>Explanation</h4>
        <ul style={styles.explanationList}>
          {result.explanation.map((line, i) => <li key={i}>{line}</li>)}
        </ul>
      </div>

      <h4 style={styles.sectionTitle}>Per-face evidence ({result.faces.length})</h4>
      {result.faces.map((face) => (
        <FaceResultCard key={face.face_id} face={face} />
      ))}

      <button style={styles.resetButton} onClick={onReset}>Analyze another file</button>
    </div>
  )
}

const styles = {
  container: { maxWidth: 720, margin: '40px auto', fontFamily: 'system-ui, sans-serif', paddingBottom: 60 },
  verdictCard: {
    border: '2px solid', borderRadius: 12, padding: 20, marginBottom: 20,
    display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#fff',
  },
  verdictLabel: { fontSize: '2rem', fontWeight: 700, letterSpacing: '0.02em' },
  confidence: { margin: '4px 0 0', color: '#555' },
  explanationBox: { background: '#f7f7f8', borderRadius: 10, padding: 16, marginBottom: 24 },
  explanationList: { margin: '8px 0 0', paddingLeft: 20, fontSize: '0.92rem', lineHeight: 1.6 },
  sectionTitle: { fontSize: '0.95rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: '#888', marginBottom: 10 },
  noFace: { textAlign: 'center', fontSize: '1.1rem', color: '#888', marginTop: 80 },
  resetButton: {
    display: 'block', margin: '20px auto 0', padding: '10px 24px', borderRadius: 8,
    border: '1px solid #ccc', background: '#fff', cursor: 'pointer', fontSize: '0.9rem',
  },
}
