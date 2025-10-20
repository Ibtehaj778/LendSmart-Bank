from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Basic logging configuration
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger("backend")

    @app.route("/health", methods=["GET"])  # simple health check
    def health() -> Any:
        logger.debug("Health check called")
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])  # main endpoint
    def predict() -> Any:
        try:
            form_data: Dict[str, Any] = request.get_json(force=True) or {}
            logger.info("/predict called - received request body")
            logger.debug("request.json=%s", safe_json_sample(form_data))
        except Exception:
            logger.exception("Invalid JSON body")
            return jsonify({"error": "Invalid JSON body"}), 400

        logger.info("Computing prediction using mock risk model")
        prediction = predict_default(form_data)
        logger.debug("prediction=%s", json.dumps(prediction, separators=(",", ":")))

        logger.info("Requesting GPT verdict (or using fallback if no API key)")
        verdict_text = generate_verdict_with_gpt(prediction, form_data)
        logger.debug("verdict=%s", verdict_text)

        prediction["verdict"] = verdict_text
        logger.info("/predict completed successfully")
        return jsonify(prediction)

    return app


def predict_default(form_data: Dict[str, Any]) -> Dict[str, Any]:
    age = to_float(form_data.get("age"), 0)
    income = to_float(form_data.get("income"), 1)  # avoid div by zero
    loan_amount = to_float(form_data.get("loan_amount"), 0)
    credit_score = to_float(form_data.get("credit_score"), 650)
    months_employed = to_float(form_data.get("months_employed"), 0)
    num_credit_lines = to_float(form_data.get("num_credit_lines"), 0)
    interest_rate = to_float(form_data.get("interest_rate"), 0)
    loan_term = to_float(form_data.get("loan_term"), 0)
    debt_to_income = to_float(form_data.get("debt_to_income"), 0)
    payment_history_score = to_float(form_data.get("payment_history_score"), 50)
    
    # New categorical fields
    education = form_data.get("education", "Bachelor's")
    employment_type = form_data.get("employment_type", "Full-time")
    marital_status = form_data.get("marital_status", "Married")
    has_mortgage = form_data.get("has_mortgage", "No")
    has_dependents = form_data.get("has_dependents", "No")
    loan_purpose = form_data.get("loan_purpose", "Home")
    has_cosigner = form_data.get("has_cosigner", "No")

    risk_score = 0.5

    # Credit score impact (negative = lower risk)
    risk_score -= (credit_score - 650.0) / 1000.0

    # Debt-to-income impact (positive = higher risk)
    risk_score += debt_to_income * 0.3

    # Payment history impact (negative = lower risk)
    risk_score -= (payment_history_score - 50.0) / 500.0

    # Employment months impact (negative = lower risk)
    employment_years = months_employed / 12.0
    risk_score -= min(employment_years / 20.0, 0.2)

    # Age impact (slight positive for very young)
    if age < 25:
        risk_score += 0.1

    # Loan-to-income ratio
    loan_to_income = loan_amount / income
    risk_score += min(loan_to_income / 5.0, 0.2)
    
    # Interest rate impact (higher rate = higher risk)
    risk_score += (interest_rate - 5.0) / 100.0
    
    # Loan term impact (longer term = slightly higher risk)
    risk_score += min(loan_term / 1000.0, 0.1)
    
    # Number of credit lines impact (more lines = slightly higher risk)
    risk_score += (num_credit_lines - 3.0) * 0.01

    # Categorical field impacts
    # Education impact (higher education = lower risk)
    education_impact = {"High School": 0.1, "Bachelor's": 0.0, "Master's": -0.05, "PhD": -0.1}
    risk_score += education_impact.get(education, 0.0)
    
    # Employment type impact
    employment_impact = {"Full-time": -0.05, "Part-time": 0.05, "Self-employed": 0.1, "Unemployed": 0.2}
    risk_score += employment_impact.get(employment_type, 0.0)
    
    # Marital status impact
    marital_impact = {"Single": 0.05, "Married": -0.05, "Divorced": 0.1}
    risk_score += marital_impact.get(marital_status, 0.0)
    
    # Mortgage impact (having mortgage = slightly higher risk due to existing debt)
    if has_mortgage == "Yes":
        risk_score += 0.05
    
    # Dependents impact (dependents = higher financial responsibility)
    if has_dependents == "Yes":
        risk_score += 0.03
    
    # Loan purpose impact
    purpose_impact = {"Home": -0.05, "Business": 0.1, "Education": 0.0, "Auto": 0.05, "Other": 0.08}
    risk_score += purpose_impact.get(loan_purpose, 0.0)
    
    # Co-signer impact (co-signer = lower risk)
    if has_cosigner == "Yes":
        risk_score -= 0.1

    # Clamp between 0 and 1
    default_probability = max(0.0, min(1.0, risk_score))
    predicted_class = 1 if default_probability > 0.5 else 0

    # SHAP-like contributions
    contributions: List[Dict[str, Any]] = [
        {
            "variable": "Credit Score",
            "contribution": -(credit_score - 650.0) / 1000.0,
            "confidence": 0.95,
            "description": "Strong predictor of default risk",
        },
        {
            "variable": "Debt-to-Income Ratio",
            "contribution": debt_to_income * 0.3,
            "confidence": 0.92,
            "description": "Indicates repayment capacity",
        },
        {
            "variable": "Payment History",
            "contribution": -(payment_history_score - 50.0) / 500.0,
            "confidence": 0.88,
            "description": "Past payment behavior",
        },
        {
            "variable": "Months Employed",
            "contribution": -min(employment_years / 20.0, 0.2),
            "confidence": 0.85,
            "description": "Job stability indicator",
        },
        {
            "variable": "Loan-to-Income Ratio",
            "contribution": min(loan_to_income / 5.0, 0.2),
            "confidence": 0.82,
            "description": "Loan size relative to income",
        },
        {
            "variable": "Age",
            "contribution": 0.1 if age < 25 else -0.02,
            "confidence": 0.65,
            "description": "Age-related risk factors",
        },
        {
            "variable": "Number of Credit Lines",
            "contribution": (num_credit_lines - 3.0) * 0.01,
            "confidence": 0.58,
            "description": "Credit portfolio diversity",
        },
        {
            "variable": "Interest Rate",
            "contribution": (interest_rate - 5.0) / 100.0,
            "confidence": 0.70,
            "description": "Loan interest rate",
        },
        {
            "variable": "Loan Term",
            "contribution": min(loan_term / 1000.0, 0.1),
            "confidence": 0.65,
            "description": "Loan duration in months",
        },
        {
            "variable": "Annual Income",
            "contribution": -(income - 40000.0) / 100000.0,
            "confidence": 0.72,
            "description": "Income level and stability",
        },
        {
            "variable": "Education Level",
            "contribution": education_impact.get(education, 0.0),
            "confidence": 0.68,
            "description": f"Educational background: {education}",
        },
        {
            "variable": "Employment Type",
            "contribution": employment_impact.get(employment_type, 0.0),
            "confidence": 0.75,
            "description": f"Employment status: {employment_type}",
        },
        {
            "variable": "Marital Status",
            "contribution": marital_impact.get(marital_status, 0.0),
            "confidence": 0.62,
            "description": f"Marital status: {marital_status}",
        },
        {
            "variable": "Mortgage Status",
            "contribution": 0.05 if has_mortgage == "Yes" else 0.0,
            "confidence": 0.58,
            "description": f"Existing mortgage: {has_mortgage}",
        },
        {
            "variable": "Dependents",
            "contribution": 0.03 if has_dependents == "Yes" else 0.0,
            "confidence": 0.55,
            "description": f"Has dependents: {has_dependents}",
        },
        {
            "variable": "Loan Purpose",
            "contribution": purpose_impact.get(loan_purpose, 0.0),
            "confidence": 0.65,
            "description": f"Loan purpose: {loan_purpose}",
        },
        {
            "variable": "Co-signer",
            "contribution": -0.1 if has_cosigner == "Yes" else 0.0,
            "confidence": 0.70,
            "description": f"Has co-signer: {has_cosigner}",
        },
    ]

    key_reasons_sorted = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
    key_reason = f"{key_reasons_sorted[0]['variable']} is the primary factor (" \
                 f"{'increasing' if key_reasons_sorted[0]['contribution'] > 0 else 'decreasing'} default risk)"

    result = {
        "default_probability": default_probability,
        "predicted_class": predicted_class,
        "key_reason": key_reason,
        "feature_contributions": contributions,
    }
    logging.getLogger("backend").debug("predict_default result=%s", json.dumps(result, separators=(",", ":")))
    return result


def generate_verdict_with_gpt(prediction: Dict[str, Any], form_data: Dict[str, Any]) -> str:
    """Use OpenAI GPT to generate a verdict using the full prediction results,
    including explainability and feature contributions. Fallback to deterministic
    verdict if no API key or if an error occurs.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    default_probability = float(prediction.get("default_probability", 0.0))
    logger = logging.getLogger("backend")
    print("API KEY LOADED:", bool(os.environ.get("OPENAI_API_KEY")))
    if not api_key:
        logger.warning("OPENAI_API_KEY not set - using deterministic verdict fallback")
        # Fallback text close to the frontend mock
        risk_level = "high" if default_probability > 0.7 else ("medium" if default_probability > 0.4 else "low")
        top_factors = ", ".join(
            [c["variable"] for c in sorted(prediction.get("feature_contributions", []), key=lambda x: abs(x["contribution"]), reverse=True)[:3]]
        )
        templates = {
            "high": f"This applicant presents a HIGH risk of default. The analysis reveals that {top_factors} are the primary concerns. With a {default_probability * 100:.1f}% probability of default, we recommend either declining this application or requiring additional collateral and stricter terms.",
            "medium": f"This applicant presents a MEDIUM risk of default. While there are some concerns around {top_factors}, the overall profile suggests manageable risk. A {default_probability * 100:.1f}% default probability indicates this loan could be approved with standard terms and regular monitoring.",
            "low": f"This applicant presents a LOW risk of default. The strong performance across {top_factors} indicates a reliable borrower. With only a {default_probability * 100:.1f}% probability of default, this application is recommended for approval with favorable terms.",
        }
        return templates[risk_level]

    # Use OpenAI API if available
    try:
        # Lazy import to avoid hard dependency if not needed
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        risk_level = "high" if default_probability > 0.7 else ("medium" if default_probability > 0.4 else "low")
        top_factors = ", ".join(
            [c["variable"] for c in sorted(prediction.get("feature_contributions", []), key=lambda x: abs(x["contribution"]), reverse=True)[:3]]
        )
        # Include full explainability content for the LLM
        explainability = json.dumps(
            prediction.get("feature_contributions", []),
            separators=(",", ":"),
        )
        prompt = (
            "You are a senior credit risk analyst. Use the provided prediction results, including the "
            "explainability feature contributions, to produce a clear 2-3 sentence verdict for a loan officer. "
            "State the risk level explicitly and provide an approval/decline recommendation with rationale.\n\n"
            f"Default probability: {default_probability * 100:.1f}%\n"
            f"Risk level: {risk_level.upper()}\n"
            f"Top factors (sorted): {top_factors}\n"
            f"Feature contributions (JSON): {explainability}\n"
            f"Applicant (JSON): {safe_json_sample(form_data)}\n"
        )

        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Calling OpenAI Chat Completions API")
        logger.debug("openai.model=%s api_key_present=%s", model_name, bool(api_key))
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert credit risk assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=160,
        )
        logger.info("OpenAI API responded successfully")
        logger.debug("openai.response=%s", safe_json_sample(response.model_dump()))
        content = (response.choices[0].message.content or "").strip()
        return content or "Verdict unavailable."
    except Exception:
        logger.exception("OpenAI API call failed - using deterministic fallback")
        # Safe fallback
        risk_level = "high" if default_probability > 0.7 else ("medium" if default_probability > 0.4 else "low")
        top_factors = ", ".join(
            [c["variable"] for c in sorted(prediction.get("feature_contributions", []), key=lambda x: abs(x["contribution"]), reverse=True)[:3]]
        )
        templates = {
            "high": f"This applicant presents a HIGH risk of default. The analysis reveals that {top_factors} are the primary concerns. With a {default_probability * 100:.1f}% probability of default, we recommend either declining this application or requiring additional collateral and stricter terms.",
            "medium": f"This applicant presents a MEDIUM risk of default. While there are some concerns around {top_factors}, the overall profile suggests manageable risk. A {default_probability * 100:.1f}% default probability indicates this loan could be approved with standard terms and regular monitoring.",
            "low": f"This applicant presents a LOW risk of default. The strong performance across {top_factors} indicates a reliable borrower. With only a {default_probability * 100:.1f}% probability of default, this application is recommended for approval with favorable terms.",
        }
        return templates[risk_level]


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_json_sample(obj: Dict[str, Any]) -> str:
    # compact 1-line representation limited in size for prompts
    try:
        text = json.dumps(obj, separators=(",", ":"))
        return text[:600]
    except Exception:
        return "{}"


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)


