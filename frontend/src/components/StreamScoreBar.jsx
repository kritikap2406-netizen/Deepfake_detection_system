export default function StreamScoreBar({ label, value }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.5 ? '#e05a4e' : '#3aa96b'

  return (
    <div style={styles.row}>
      <span style={styles.label}>{label}</span>
      <div style={styles.track}>
        <div style={{ ...styles.fill, width: `${pct}%`, background: color }} />
      </div>
      <span style={styles.value}>{value.toFixed(2)}</span>
    </div>
  )
}

const styles = {
  row: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 },
  label: { width: 110, fontSize: '0.82rem', color: '#555', flexShrink: 0 },
  track: { flex: 1, height: 8, background: '#eee', borderRadius: 4, overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 4, transition: 'width 0.3s' },
  value: { width: 36, fontSize: '0.8rem', color: '#333', textAlign: 'right', flexShrink: 0 },
}
