import { type NextRequest, NextResponse } from "next/server"

// Mock ML model prediction
function predictDefault(formData: any) {
  const {
    age,
    income,
    credit_score,
    loan_amount,
    employment_years,
    debt_to_income,
    payment_history_score,
    num_accounts,
  } = formData

  // Simplified mock prediction logic
  let riskScore = 0.5

  // Credit score impact (negative = lower risk)
  riskScore -= (credit_score - 650) / 1000

  // Debt-to-income impact (positive = higher risk)
  riskScore += debt_to_income * 0.3

  // Payment history impact (negative = lower risk)
  riskScore -= (payment_history_score - 50) / 500

  // Employment years impact (negative = lower risk)
  riskScore -= Math.min(employment_years / 20, 0.2)

  // Age impact (slight positive for very young)
  if (age < 25) riskScore += 0.1

  // Loan-to-income ratio
  const loanToIncome = loan_amount / income
  riskScore += Math.min(loanToIncome / 5, 0.2)

  // Clamp between 0 and 1
  const defaultProbability = Math.max(0, Math.min(1, riskScore))
  const predictedClass = defaultProbability > 0.5 ? 1 : 0

  // Generate feature contributions (SHAP-like values)
  const contributions = [
    {
      variable: "Credit Score",
      contribution: -(credit_score - 650) / 1000,
      confidence: 0.95,
      description: "Strong predictor of default risk",
    },
    {
      variable: "Debt-to-Income Ratio",
      contribution: debt_to_income * 0.3,
      confidence: 0.92,
      description: "Indicates repayment capacity",
    },
    {
      variable: "Payment History",
      contribution: -(payment_history_score - 50) / 500,
      confidence: 0.88,
      description: "Past payment behavior",
    },
    {
      variable: "Employment Years",
      contribution: -Math.min(employment_years / 20, 0.2),
      confidence: 0.85,
      description: "Job stability indicator",
    },
    {
      variable: "Loan-to-Income Ratio",
      contribution: Math.min(loanToIncome / 5, 0.2),
      confidence: 0.82,
      description: "Loan size relative to income",
    },
    {
      variable: "Age",
      contribution: age < 25 ? 0.1 : -0.02,
      confidence: 0.65,
      description: "Age-related risk factors",
    },
    {
      variable: "Number of Accounts",
      contribution: (num_accounts - 3) * 0.02,
      confidence: 0.58,
      description: "Credit portfolio diversity",
    },
    {
      variable: "Annual Income",
      contribution: -(income - 40000) / 100000,
      confidence: 0.72,
      description: "Income level and stability",
    },
  ]

  // Determine key reason
  const keyReasons = contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
  const keyReason = `${keyReasons[0].variable} is the primary factor (${
    keyReasons[0].contribution > 0 ? "increasing" : "decreasing"
  } default risk)`

  return {
    default_probability: defaultProbability,
    predicted_class: predictedClass,
    key_reason: keyReason,
    feature_contributions: contributions,
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.json()
    const prediction = predictDefault(formData)
    return NextResponse.json(prediction)
  } catch (error) {
    return NextResponse.json({ error: "Prediction failed" }, { status: 500 })
  }
}
