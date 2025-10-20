"use client"

import {
  RadarChart as RechartsRadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts"

export default function RadarChart({ contributions }) {
  const data = contributions.slice(0, 5).map((item) => ({
    name: item.variable,
    confidence: item.confidence * 100,
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RechartsRadarChart data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="name" />
        <PolarRadiusAxis angle={90} domain={[0, 100]} />
        <Radar name="Confidence Score" dataKey="confidence" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
        <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
      </RechartsRadarChart>
    </ResponsiveContainer>
  )
}
