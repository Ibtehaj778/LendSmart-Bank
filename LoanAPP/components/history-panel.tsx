"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Clock } from "lucide-react"

export default function HistoryPanel({ history, onSelect }) {
  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="w-4 h-4" />
            Recent Predictions
          </CardTitle>
          <CardDescription>Last 3 predictions</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No predictions yet. Submit a form to get started.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="w-4 h-4" />
          Recent Predictions
        </CardTitle>
        <CardDescription>Last 3 predictions</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {history.map((item, idx) => (
          <Button
            key={idx}
            variant="outline"
            className="w-full justify-between h-auto py-3 px-3 bg-transparent"
            onClick={() => onSelect(item)}
          >
            <div className="text-left">
              <div className="font-semibold text-sm">{(item.default_probability * 100).toFixed(1)}% Risk</div>
              <div className="text-xs text-muted-foreground">{new Date(item.timestamp).toLocaleString()}</div>
            </div>
            <Badge variant={item.default_probability > 0.5 ? "destructive" : "default"} className="ml-2">
              {item.predicted_class === 1 ? "Default" : "Approved"}
            </Badge>
          </Button>
        ))}
      </CardContent>
    </Card>
  )
}
