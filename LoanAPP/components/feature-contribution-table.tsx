"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { HelpCircle } from "lucide-react"

export default function FeatureContributionTable({ contributions }) {
  const sortedContributions = [...contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))

  return (
    <Card>
      <CardHeader>
        <CardTitle>Feature Contributions</CardTitle>
        <CardDescription>Variables ranked by impact on prediction (SHAP values)</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Rank</TableHead>
                <TableHead>Variable</TableHead>
                <TableHead>
                  <div className="flex items-center gap-1">
                    Contribution
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="w-4 h-4 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>SHAP value indicating impact direction and magnitude</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                </TableHead>
                <TableHead>
                  <div className="flex items-center gap-1">
                    Confidence
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="w-4 h-4 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>Model confidence in this feature's importance</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                </TableHead>
                <TableHead>Description</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sortedContributions.map((item, idx) => (
                <TableRow key={item.variable}>
                  <TableCell className="font-semibold">{idx + 1}</TableCell>
                  <TableCell className="font-medium">{item.variable}</TableCell>
                  <TableCell>
                    <Badge variant={item.contribution > 0 ? "destructive" : "default"} className="font-mono">
                      {item.contribution > 0 ? "+" : ""}
                      {item.contribution.toFixed(4)}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-muted rounded-full h-2">
                        <div className="bg-primary h-2 rounded-full" style={{ width: `${item.confidence * 100}%` }} />
                      </div>
                      <span className="text-sm font-medium">{(item.confidence * 100).toFixed(0)}%</span>
                      {item.verdict && <span className="text-xs text-muted-foreground ml-2">{item.verdict}</span>}
                    </div>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">{item.description}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}
