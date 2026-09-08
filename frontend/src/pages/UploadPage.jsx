import { useRef, useState } from 'react'
import { inferMediaType } from '../services/api'

export default function UploadPage({ onFileSelected }) {
  const inputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState(null)

  function handleFile(file) {
    if (!file) return
    const mediaType = inferMediaType(file)
    if (!mediaType) {
      setError('Unsupported file type. Please upload a JPG/PNG image or an MP4/AVI/MOV/MKV video.')
      return
    }
    setError(null)
    onFileSelected(file, mediaType)
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>TrueSight</h1>
      <p style={styles.subtitle}>Multi-stream deepfake detection — upload an image or video to analyze</p>

      <div
        style={{ ...styles.dropzone, ...(dragActive ? styles.dropzoneActive : {}) }}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragActive(false)
          handleFile(e.dataTransfer.files?.[0])
        }}
        onClick={() => inputRef.current?.click()}
      >
        <p style={{ margin: 0, fontSize: '1.1rem' }}>Drag & drop a file here, or click to browse</p>
        <p style={{ margin: '8px 0 0', fontSize: '0.85rem', color: '#888' }}>
          Images: JPG, PNG &nbsp;·&nbsp; Videos: MP4, AVI, MOV, MKV
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.mp4,.avi,.mov,.mkv"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {error && <p style={styles.error}>{error}</p>}
    </div>
  )
}

const styles = {
  container: { maxWidth: 640, margin: '60px auto', textAlign: 'center', fontFamily: 'system-ui, sans-serif' },
  title: { fontSize: '2.2rem', margin: 0, letterSpacing: '-0.02em' },
  subtitle: { color: '#666', marginTop: 8, marginBottom: 32 },
  dropzone: {
    border: '2px dashed #ccc', borderRadius: 12, padding: '48px 24px',
    cursor: 'pointer', transition: 'border-color 0.15s, background 0.15s',
  },
  dropzoneActive: { borderColor: '#4a7dff', background: '#f5f8ff' },
  error: { color: '#c0392b', marginTop: 16 },
}
