import { useState } from 'react'
import UploadPage from './pages/UploadPage'
import ProcessingScreen from './pages/ProcessingScreen'
import ResultsDashboard from './pages/ResultsDashboard'
import { analyzeMedia } from './services/api'

// Simple explicit state machine: upload -> processing -> results | error.
// No routing library needed for a 3-screen prototype flow.
export default function App() {
  const [stage, setStage] = useState('upload')
  const [pendingFile, setPendingFile] = useState(null)
  const [mediaType, setMediaType] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleFileSelected(file, inferredMediaType) {
    setPendingFile(file)
    setMediaType(inferredMediaType)
    setStage('processing')
    setError(null)

    try {
      const data = await analyzeMedia(file)
      setResult(data)
      setStage('results')
    } catch (err) {
      setError(err.message)
      setStage('upload')
    }
  }

  function reset() {
    setStage('upload')
    setPendingFile(null)
    setMediaType(null)
    setResult(null)
    setError(null)
  }

  if (stage === 'processing') {
    return <ProcessingScreen fileName={pendingFile?.name} mediaType={mediaType} />
  }

  if (stage === 'results' && result) {
    return <ResultsDashboard result={result} onReset={reset} />
  }

  return (
    <div>
      <UploadPage onFileSelected={handleFileSelected} />
      {error && (
        <p style={{ textAlign: 'center', color: '#c0392b', fontFamily: 'system-ui, sans-serif' }}>
          {error}
        </p>
      )}
    </div>
  )
}
