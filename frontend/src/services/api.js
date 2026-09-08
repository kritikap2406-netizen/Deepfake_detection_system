// TrueSight — API service layer.
// Talks to the real FastAPI backend built in Phase 14. No mocking —
// these functions call the actual /analyze/image, /analyze/video,
// /result/{id}, and /health endpoints.

const IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']
const VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv']

export function inferMediaType(file) {
  const ext = file.name.split('.').pop().toLowerCase()
  if (IMAGE_EXTENSIONS.includes(ext)) return 'image'
  if (VIDEO_EXTENSIONS.includes(ext)) return 'video'
  return null
}

async function parseErrorResponse(response) {
  try {
    const data = await response.json()
    return data.detail || `Request failed with status ${response.status}`
  } catch {
    return `Request failed with status ${response.status}`
  }
}

export async function analyzeMedia(file) {
  const mediaType = inferMediaType(file)
  if (!mediaType) {
    throw new Error(
      `Unsupported file type. Allowed: ${[...IMAGE_EXTENSIONS, ...VIDEO_EXTENSIONS].join(', ')}`
    )
  }

  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`/analyze/${mediaType}`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response))
  }

  return response.json()
}

export async function fetchResult(resultId) {
  const response = await fetch(`/result/${resultId}`)
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response))
  }
  return response.json()
}

export async function fetchHealth() {
  const response = await fetch('/health')
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response))
  }
  return response.json()
}
