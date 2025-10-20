"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import BarChart from "./charts/bar-chart"
import RadarChart from "./charts/radar-chart"
import GaugeChart from "./charts/gauge-chart"

export default function VisualizationPanel({ contributions }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Visualizations</CardTitle>
        <CardDescription>Interactive charts showing feature importance and risk assessment</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="bar" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="bar">Feature Impact</TabsTrigger>
            <TabsTrigger value="radar">Confidence Radar</TabsTrigger>
            <TabsTrigger value="gauge">Risk Gauge</TabsTrigger>
          </TabsList>

          <TabsContent value="bar" className="mt-6">
            <BarChart contributions={contributions} />
          </TabsContent>

          <TabsContent value="radar" className="mt-6">
            <RadarChart contributions={contributions} />
          </TabsContent>

          <TabsContent value="gauge" className="mt-6">
            <GaugeChart />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
