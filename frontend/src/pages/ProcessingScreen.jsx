export default function ProcessingScreen({ fileName, mediaType }) {
  return (
    <div style={styles.container}>
      <div style={styles.spinner} />
      <p style={styles.text}>
        Analyzing {mediaType === 'video' ? 'video' : 'image'}: <strong>{fileName}</strong>
      </p>
      <p style={styles.subtext}>
        Running spatial, frequency, noise{mediaType === 'video' ? ', physiological, and temporal' : ''} analysis…
      </p>
    </div>
  )
}

const styles = {
  container: { maxWidth: 640, margin: '100px auto', textAlign: 'center', fontFamily: 'system-ui, sans-serif' },
  spinner: {
    width: 40, height: 40, margin: '0 auto 20px', borderRadius: '50%',
    border: '4px solid #eee', borderTopColor: '#4a7dff',
    animation: 'truesight-spin 0.8s linear infinite',
  },
  text: { fontSize: '1.1rem' },
  subtext: { color: '#888', fontSize: '0.9rem' },
}

// Injects the keyframes once (avoids needing a separate CSS file for one animation).
if (typeof document !== 'undefined' && !document.getElementById('truesight-spin-style')) {
  const style = document.createElement('style')
  style.id = 'truesight-spin-style'
  style.textContent = '@keyframes truesight-spin { to { transform: rotate(360deg); } }'
  document.head.appendChild(style)
}
