"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Sparkles } from "lucide-react"

// Function to clean markdown formatting from text
function cleanMarkdownText(text: string): string {
  if (!text) return text
  
  return text
    // Remove code blocks (```...```)
    .replace(/```[\s\S]*?```/g, '')
    // Remove inline code (`...`)
    .replace(/`([^`]+)`/g, '$1')
    // Remove markdown headers (# ## ###)
    .replace(/^#{1,6}\s+/gm, '')
    // Remove bold/italic (**text** or *text*)
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    // Remove markdown links [text](url)
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Clean up extra whitespace
    .replace(/\n\s*\n/g, '\n')
    .trim()
}

export default function VerdictBox({ verdict }) {
  const cleanedVerdict = cleanMarkdownText(verdict)
  
  return (
    <Card className="border-2 border-primary/20 bg-primary/5">
      <CardContent className="pt-6">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <Badge variant="outline" className="gap-1 font-bold">
              <Sparkles className="w-3 h-3" />
              AI Verdict
            </Badge>
          </div>
          <div className="flex-1">
            <div className="bg-white dark:bg-slate-950 rounded-lg p-4 border border-border">
              <p className="text-foreground leading-relaxed">{cleanedVerdict}</p>
            </div>
            <p className="text-xs text-muted-foreground mt-2 italic">
              Generated via SHAP interpretation and NLP reasoning using LendSmart Explainability API.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
