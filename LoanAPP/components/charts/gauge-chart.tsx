"use client"

import { useEffect, useRef } from "react"

export default function GaugeChart() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) / 2 - 20

    // Clear canvas
    ctx.fillStyle = "#f5f5f5"
    ctx.fillRect(0, 0, width, height)

    // Draw gauge background
    ctx.strokeStyle = "#e5e7eb"
    ctx.lineWidth = 20
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, Math.PI, 0, false)
    ctx.stroke()

    // Draw colored segments
    const segments = [
      { color: "#22c55e", start: Math.PI, end: Math.PI + Math.PI / 3 }, // Green
      { color: "#eab308", start: Math.PI + Math.PI / 3, end: Math.PI + (2 * Math.PI) / 3 }, // Yellow
      { color: "#ef4444", start: Math.PI + (2 * Math.PI) / 3, end: 2 * Math.PI }, // Red
    ]

    segments.forEach((segment) => {
      ctx.strokeStyle = segment.color
      ctx.lineWidth = 20
      ctx.beginPath()
      ctx.arc(centerX, centerY, radius, segment.start, segment.end, false)
      ctx.stroke()
    })

    // Draw needle (pointing to 65% - medium risk)
    const angle = Math.PI + Math.PI * 0.65
    const needleLength = radius - 10
    ctx.strokeStyle = "#1f2937"
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(centerX, centerY)
    ctx.lineTo(centerX + needleLength * Math.cos(angle), centerY + needleLength * Math.sin(angle))
    ctx.stroke()

    // Draw center circle
    ctx.fillStyle = "#1f2937"
    ctx.beginPath()
    ctx.arc(centerX, centerY, 8, 0, 2 * Math.PI)
    ctx.fill()

    // Draw labels
    ctx.fillStyle = "#6b7280"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Low", centerX - radius - 10, centerY + 5)
    ctx.fillText("Medium", centerX, centerY + radius + 20)
    ctx.fillText("High", centerX + radius + 10, centerY + 5)
  }, [])

  return (
    <div className="flex justify-center">
      <canvas ref={canvasRef} width={300} height={200} className="border border-border rounded-lg" />
    </div>
  )
}
