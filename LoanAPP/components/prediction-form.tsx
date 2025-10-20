"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { generatePDF } from "@/lib/pdf-generator"

export default function PredictionForm({ onSubmit, loading, prediction }) {
  const [formData, setFormData] = useState({
    applicant_name: "",
    age: 35,
    income: 50000,
    loan_amount: 200000,
    credit_score: 700,
    months_employed: 60,
    num_credit_lines: 4,
    interest_rate: 5.5,
    loan_term: 360,
    debt_to_income: 0.35,
    payment_history_score: 85,
    // New categorical fields
    education: "Bachelor's",
    employment_type: "Full-time",
    marital_status: "Married",
    has_mortgage: "Yes",
    has_dependents: "Yes",
    loan_purpose: "Home",
    has_cosigner: "No",
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: Number.parseFloat(value) || value,
    }))
  }

  const handleSelectChange = (name, value) => {
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(formData)
  }

  const handleDownloadReport = async () => {
    if (prediction) {
      await generatePDF(prediction)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2 md:col-span-2">
          <Label htmlFor="applicant_name">Applicant Name</Label>
          <Input id="applicant_name" name="applicant_name" type="text" value={formData.applicant_name} onChange={handleChange} placeholder="e.g., Jane Doe" />
        </div>
        <div className="space-y-2">
          <Label htmlFor="age">Age</Label>
          <Input id="age" name="age" type="number" value={formData.age} onChange={handleChange} min="18" max="100" />
        </div>

        <div className="space-y-2">
          <Label htmlFor="income">Income ($)</Label>
          <Input
            id="income"
            name="income"
            type="number"
            value={formData.income}
            onChange={handleChange}
            min="0"
            step="1000"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="loan_amount">Loan Amount ($)</Label>
          <Input
            id="loan_amount"
            name="loan_amount"
            type="number"
            value={formData.loan_amount}
            onChange={handleChange}
            min="0"
            step="10000"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="credit_score">Credit Score</Label>
          <Input
            id="credit_score"
            name="credit_score"
            type="number"
            value={formData.credit_score}
            onChange={handleChange}
            min="300"
            max="850"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="months_employed">Months Employed</Label>
          <Input
            id="months_employed"
            name="months_employed"
            type="number"
            value={formData.months_employed}
            onChange={handleChange}
            min="0"
            max="720"
            step="1"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="num_credit_lines">Number of Credit Lines</Label>
          <Input
            id="num_credit_lines"
            name="num_credit_lines"
            type="number"
            value={formData.num_credit_lines}
            onChange={handleChange}
            min="0"
            max="20"
            step="1"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="interest_rate">Interest Rate (%)</Label>
          <Input
            id="interest_rate"
            name="interest_rate"
            type="number"
            value={formData.interest_rate}
            onChange={handleChange}
            min="0"
            max="30"
            step="0.1"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="loan_term">Loan Term (Months)</Label>
          <Input
            id="loan_term"
            name="loan_term"
            type="number"
            value={formData.loan_term}
            onChange={handleChange}
            min="1"
            max="480"
            step="1"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="debt_to_income">Debt-to-Income Ratio</Label>
          <Input
            id="debt_to_income"
            name="debt_to_income"
            type="number"
            value={formData.debt_to_income}
            onChange={handleChange}
            min="0"
            max="1"
            step="0.01"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="payment_history_score">Payment History Score</Label>
          <Input
            id="payment_history_score"
            name="payment_history_score"
            type="number"
            value={formData.payment_history_score}
            onChange={handleChange}
            min="0"
            max="100"
          />
        </div>


        {/* New Categorical Fields */}
        <div className="space-y-2">
          <Label htmlFor="education">Education Level</Label>
          <Select value={formData.education} onValueChange={(value) => handleSelectChange("education", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select education level" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="High School">High School</SelectItem>
              <SelectItem value="Bachelor's">Bachelor's</SelectItem>
              <SelectItem value="Master's">Master's</SelectItem>
              <SelectItem value="PhD">PhD</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="employment_type">Employment Type</Label>
          <Select value={formData.employment_type} onValueChange={(value) => handleSelectChange("employment_type", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select employment type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Full-time">Full-time</SelectItem>
              <SelectItem value="Part-time">Part-time</SelectItem>
              <SelectItem value="Self-employed">Self-employed</SelectItem>
              <SelectItem value="Unemployed">Unemployed</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="marital_status">Marital Status</Label>
          <Select value={formData.marital_status} onValueChange={(value) => handleSelectChange("marital_status", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select marital status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Single">Single</SelectItem>
              <SelectItem value="Married">Married</SelectItem>
              <SelectItem value="Divorced">Divorced</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="has_mortgage">Has Mortgage</Label>
          <Select value={formData.has_mortgage} onValueChange={(value) => handleSelectChange("has_mortgage", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select mortgage status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Yes">Yes</SelectItem>
              <SelectItem value="No">No</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="has_dependents">Has Dependents</Label>
          <Select value={formData.has_dependents} onValueChange={(value) => handleSelectChange("has_dependents", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select dependents status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Yes">Yes</SelectItem>
              <SelectItem value="No">No</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="loan_purpose">Loan Purpose</Label>
          <Select value={formData.loan_purpose} onValueChange={(value) => handleSelectChange("loan_purpose", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select loan purpose" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Home">Home</SelectItem>
              <SelectItem value="Business">Business</SelectItem>
              <SelectItem value="Education">Education</SelectItem>
              <SelectItem value="Auto">Auto</SelectItem>
              <SelectItem value="Other">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="has_cosigner">Has Co-signer</Label>
          <Select value={formData.has_cosigner} onValueChange={(value) => handleSelectChange("has_cosigner", value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select co-signer status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Yes">Yes</SelectItem>
              <SelectItem value="No">No</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex gap-2">
        <Button type="submit" disabled={loading} className="flex-1">
          {loading ? "Generating Prediction..." : "Generate Prediction"}
        </Button>
      </div>
    </form>
  )
}
