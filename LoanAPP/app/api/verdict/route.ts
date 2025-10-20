import { type NextRequest, NextResponse } from "next/server"

// Mock NLP verdict generation
function generateVerdictText(prediction: any, formData: any) {
  const { default_probability, feature_contributions, key_reason } = prediction
  const riskLevel = default_probability > 0.7 ? "high" : default_probability > 0.4 ? "medium" : "low"

  const topFactors = feature_contributions
    .sort((a: any, b: any) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 3)
    .map((f: any) => f.variable)
    .join(", ")

  const verdicts = {
    high: `This applicant presents a HIGH risk of default. The analysis reveals that ${topFactors} are the primary concerns. With a ${(default_probability * 100).toFixed(1)}% probability of default, we recommend either declining this application or requiring additional collateral and stricter terms.`,
    medium: `This applicant presents a MEDIUM risk of default. While there are some concerns around ${topFactors}, the overall profile suggests manageable risk. A ${(default_probability * 100).toFixed(1)}% default probability indicates this loan could be approved with standard terms and regular monitoring.`,
    low: `This applicant presents a LOW risk of default. The strong performance across ${topFactors} indicates a reliable borrower. With only a ${(default_probability * 100).toFixed(1)}% probability of default, this application is recommended for approval with favorable terms.`,
  }

  return verdicts[riskLevel as keyof typeof verdicts]
}

export async function POST(request: NextRequest) {
  try {
    const { prediction, formData } = await request.json()
    const verdict = generateVerdictText(prediction, formData)
    return NextResponse.json({ verdict })
  } catch (error) {
    return NextResponse.json({ error: "Verdict generation failed" }, { status: 500 })
  }
}
