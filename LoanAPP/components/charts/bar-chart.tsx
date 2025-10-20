"use client"

import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

export default function BarChart({ contributions }) {
  const data = contributions
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8)
    .map((item) => ({
      name: item.variable,
      contribution: item.contribution,
      fill: item.contribution > 0 ? "#ef4444" : "#3b82f6",
    }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RechartsBarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
        <YAxis />
        <Tooltip formatter={(value) => value.toFixed(4)} labelFormatter={(label) => `${label}`} />
        <Bar dataKey="contribution" fill="#8884d8" />
      </RechartsBarChart>
    </ResponsiveContainer>
  )
}
