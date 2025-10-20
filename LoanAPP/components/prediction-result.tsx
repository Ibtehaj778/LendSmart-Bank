"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertCircle, Download } from "lucide-react"
import FeatureContributionTable from "./feature-contribution-table"
import VisualizationPanel from "./visualization-panel"
import VerdictBox from "./verdict-box"
import { generatePDF } from "@/lib/pdf-generator"

export default function PredictionResult({
  prediction,
  predictionRef,
  verdictRef,
  visualizationRef,
  featureContributionsRef,
}) {
  const [showDetails, setShowDetails] = useState(true)

  const defaultProbability = prediction.default_probability
  const isHighRisk = defaultProbability > 0.5
  const riskLevel = defaultProbability > 0.7 ? "High" : defaultProbability > 0.4 ? "Medium" : "Low"

  const handleDownloadReport = async () => {
    await generatePDF(prediction)
  }

  return (
    <div className="space-y-6">
      {/* Main Prediction Score */}
      <div ref={predictionRef} className="scroll-mt-20">
        <Card className={`border-2 ${isHighRisk ? "border-destructive" : "border-green-500"}`}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="font-bold">Prediction Result</CardTitle>
                <CardDescription>Default Risk Assessment</CardDescription>
              </div>
              <div className="flex items-center gap-3">
                <Badge variant={isHighRisk ? "destructive" : "default"}>{riskLevel} Risk</Badge>
                <Button onClick={handleDownloadReport} variant="outline" size="sm" className="gap-2 bg-transparent">
                  <Download className="w-4 h-4" />
                  Download Report
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-6xl font-bold text-foreground">{(defaultProbability * 100).toFixed(1)}%</div>
                <p className="text-muted-foreground mt-2 italic">Probability of Default</p>
              </div>
              <div className="flex flex-col items-center gap-4">
                {/* Status Badge */}
                <div className={`px-6 py-3 rounded-lg font-bold text-lg ${
                  prediction.predicted_class === 1 
                    ? "bg-red-100 text-red-800 border-2 border-red-300" 
                    : "bg-green-100 text-green-800 border-2 border-green-300"
                }`}>
                  {prediction.predicted_class === 1 ? "❌ DECLINED" : "✅ APPROVED"}
                </div>
                
                {/* Decision Circle */}
                <div className={`w-24 h-24 rounded-full flex items-center justify-center ${
                  prediction.predicted_class === 1 
                    ? "bg-gradient-to-br from-red-500 to-red-600" 
                    : "bg-gradient-to-br from-green-500 to-green-600"
                }`}>
                  <div className="text-white font-bold text-sm text-center">
                    {prediction.predicted_class === 1 ? "HIGH RISK" : "LOW RISK"}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {prediction.verdict && (
        <div ref={verdictRef} className="scroll-mt-20">
          <VerdictBox verdict={prediction.verdict} />
        </div>
      )}

      {/* Key Reason */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 font-bold">
            <AlertCircle className="w-5 h-5" />
            Key Reason for Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-lg font-semibold text-foreground">{prediction.key_reason}</p>
          <p className="text-muted-foreground mt-2 italic">
            This is the most influential factor in the model's decision
          </p>
        </CardContent>
      </Card>

      <div ref={visualizationRef} className="scroll-mt-20">
        <VisualizationPanel contributions={prediction.feature_contributions} />
      </div>

      <div ref={featureContributionsRef} className="scroll-mt-20">
        <FeatureContributionTable contributions={prediction.feature_contributions} />
      </div>
    </div>
  )
}
