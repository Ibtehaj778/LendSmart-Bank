"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/hooks/use-toast"
import PredictionForm from "@/components/prediction-form"
import PredictionResult from "@/components/prediction-result"
import HistoryPanel from "@/components/history-panel"

export default function Home() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState([])
  const [activeSection, setActiveSection] = useState("prediction")
  const { toast } = useToast()

  const predictionRef = useRef(null)
  const verdictRef = useRef(null)
  const visualizationRef = useRef(null)
  const featureContributionsRef = useRef(null)

  // Load history from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("predictionHistory")
    if (saved) {
      try {
        setHistory(JSON.parse(saved))
      } catch (e) {
        console.error("Failed to load history:", e)
      }
    }
  }, [])

  useEffect(() => {
    const handleScroll = () => {
      const sections = [
        { id: "prediction", ref: predictionRef },
        { id: "verdict", ref: verdictRef },
        { id: "visualization", ref: visualizationRef },
        { id: "feature-contributions", ref: featureContributionsRef },
      ]

      for (const section of sections) {
        if (section.ref.current) {
          const rect = section.ref.current.getBoundingClientRect()
          if (rect.top <= 150) {
            setActiveSection(section.id)
          }
        }
      }
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  const handlePrediction = async (formData) => {
    setLoading(true)
    try {
      const backendBase = process.env.NEXT_PUBLIC_BACKEND_URL
      let data
      if (backendBase) {
        const res = await fetch(`${backendBase.replace(/\/$/, "")}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        })
        if (!res.ok) throw new Error("Prediction failed")
        data = await res.json()
        // propagate identifiers for PDF and UI
        data.applicant_name = formData.applicant_name
      } else {
        // Fallback to Next.js internal API routes if backend is not configured
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        })
        if (!response.ok) throw new Error("Prediction failed")
        data = await response.json()
        data.applicant_name = formData.applicant_name

        // Get NLP verdict from internal endpoint
        const verdictResponse = await fetch("/api/verdict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prediction: data, formData }),
        })
        if (verdictResponse.ok) {
          const verdictData = await verdictResponse.json()
          data.verdict = verdictData.verdict
        }
      }

      setPrediction(data)

      // Save to history (keep last 3)
      const newHistory = [{ ...data, timestamp: new Date().toISOString() }, ...history].slice(0, 3)
      setHistory(newHistory)
      localStorage.setItem("predictionHistory", JSON.stringify(newHistory))

      toast({
        title: "Prediction Complete",
        description: `Default probability: ${(data.default_probability * 100).toFixed(1)}%`,
      })

      setTimeout(() => {
        if (verdictRef.current) {
          verdictRef.current.scrollIntoView({ behavior: "smooth", block: "start" })
          setActiveSection("verdict")
        }
      }, 300)
    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const handleLoadHistory = (item) => {
    setPrediction(item)
  }

  const scrollToSection = (sectionId) => {
    const refs = {
      prediction: predictionRef,
      verdict: verdictRef,
      visualization: visualizationRef,
      "feature-contributions": featureContributionsRef,
    }

    if (refs[sectionId]?.current) {
      refs[sectionId].current.scrollIntoView({ behavior: "smooth", block: "start" })
      setActiveSection(sectionId)
    }
  }

  return (
    <>
      <header className="sticky top-0 z-50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-border">
        <div className="max-w-7xl mx-auto px-4 md:px-8">
          <div className="flex items-center justify-between h-16">
            <h1 className="text-xl font-bold">LendSmart Bank AI Credit Analyzer</h1>
            <nav className="hidden md:flex items-center gap-8">
              {[
                { id: "prediction", label: "Prediction" },
                { id: "verdict", label: "Verdict" },
                { id: "visualization", label: "Visualization" },
                { id: "feature-contributions", label: "Feature Contributions" },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className={`text-sm font-medium transition-colors ${
                    activeSection === item.id
                      ? "text-primary border-b-2 border-primary pb-1"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main className="min-h-screen bg-gradient-to-br from-background to-muted p-4 md:p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8 text-center">
            <h2 className="text-4xl font-bold text-foreground mb-2">LendSmart Bank AI Credit Analyzer</h2>
            <p className="text-muted-foreground italic">AI-powered credit analysis with explainability insights</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Main Content */}
            <div className="lg:col-span-3 space-y-6">
              {/* Input Form */}
              <div>
                <Card>
                  <CardHeader>
                    <CardTitle>Applicant Information</CardTitle>
                    <CardDescription>Enter applicant details to generate prediction</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <PredictionForm onSubmit={handlePrediction} loading={loading} />
                  </CardContent>
                </Card>
              </div>

              {/* Results */}
              {prediction && (
                <>
                  <PredictionResult
                    prediction={prediction}
                    predictionRef={predictionRef}
                    verdictRef={verdictRef}
                    visualizationRef={visualizationRef}
                    featureContributionsRef={featureContributionsRef}
                  />
                </>
              )}
            </div>

            {/* Sidebar - History */}
            <div className="lg:col-span-1">
              <HistoryPanel history={history} onSelect={handleLoadHistory} />
            </div>
          </div>
        </div>
      </main>
    </>
  )
}
