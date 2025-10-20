import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap

from flask import Flask, request, jsonify
from flask_cors import CORS


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib
from lightgbm import LGBMClassifier  # If using LightGBM
from sklearn.metrics import accuracy_score, precision_score, recall_score
import shap


class LoanFeatureEngineer:
    def __init__(self):
        self.edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
        self.medians = None
        self.columns = None

    def fit(self, df):
        DF = self._engineer_features(df.copy())
        self.medians = DF.median(numeric_only=True)
        self.columns = DF.columns.tolist()
        return DF

    def transform(self, df):
        DF = self._engineer_features(df.copy())
        DF = DF.fillna(self.medians)
        if self.columns is not None:
            DF = DF.reindex(columns=self.columns, fill_value=0)
        return DF

    def save(self, path):
        joblib.dump({'medians': self.medians, 'columns': self.columns}, path)

    def load(self, path):
        state = joblib.load(path)
        self.medians = state['medians']
        self.columns = state['columns']

    def _engineer_features(self, df):
        rng = np.random.default_rng(42)
        # ========== NUMERIC DERIVATIVES ==========
        df["LoanToIncomeRatio"] = (df["LoanAmount"] / df["Income"]).replace([np.inf, -np.inf], np.nan)
        df["DebtBurdenIndex"] = df["DTIRatio"] * df["LoanToIncomeRatio"]
        df["EffectiveInterestExposure"] = (df["InterestRate"] * df["LoanTerm"]) / 12
        df["RiskAdjustedLoan"] = df["LoanAmount"] * (1 / df["CreditScore"].replace(0, np.nan))
        df["CreditUtilizationRatio"] = df["LoanAmount"] / (df["NumCreditLines"] * df["Income"] / 12 + 1e-6)
        df["EmploymentStabilityScore"] = np.log1p(df["MonthsEmployed"])
        df["CreditLinesPerYear"] = df["NumCreditLines"] / ((df["MonthsEmployed"] / 12) + 1)
        df["CareerContinuityIndex"] = (df["MonthsEmployed"] / (df["LoanTerm"] + 1)) * 10
        df["IncomeStabilityIndex"] = (1 / (1 + df["DTIRatio"])) * np.log1p(df["MonthsEmployed"])
        df["FinancialHealthIndex"] = (
            (df["CreditScore"] / 850) * 0.5 +
            (1 - df["LoanToIncomeRatio"].clip(0, 2)) * 0.3 +
            (1 - df["DTIRatio"].clip(0, 1.5)) * 0.2
        )
        df["BorrowingPressureScore"] = (
            (df["InterestRate"] / 100) * 0.4 +
            (df["LoanTerm"] / 60) * 0.3 +
            (df["DTIRatio"].clip(0, 2)) * 0.3
        )
        df["CreditMaturityIndex"] = (df["LoanTerm"] / 12) / (df["NumCreditLines"] + 1)
        df["IncomeCreditScoreInteraction"] = df["Income"] * df["CreditScore"]
        df["LoanTermByInterest"] = df["LoanTerm"] * df["InterestRate"]
        df["CombinedDebtIndex"] = df["DTIRatio"] * df["LoanToIncomeRatio"]

        # ========== CATEGORICAL/ORDINAL SIGNALS ==========
        if "Education" in df.columns and df["Education"].dtype == object:
            df["EducationLevelIndex"] = df["Education"].map(self.edu_map).fillna(0).astype(int)
        else:
            df["EducationLevelIndex"] = 0

        def _flag(col, value):
            return (df[col] == value).astype(float) if (col in df.columns and df[col].dtype == object) else 0.0
        is_full_time = _flag("EmploymentType", "Full-time")
        is_self_emp  = _flag("EmploymentType", "Self-employed")
        is_unemp     = _flag("EmploymentType", "Unemployed")
        is_part_time = _flag("EmploymentType", "Part-time")
class LoanFeatureEngineer:
    def __init__(self):
        self.edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
        self.medians = None
        self.columns = None

    def fit(self, df):
        DF = self._engineer_features(df.copy())
        self.medians = DF.median(numeric_only=True)
        self.columns = DF.columns.tolist()
        return DF

    def transform(self, df):
        DF = self._engineer_features(df.copy())
        DF = DF.fillna(self.medians)
        if self.columns is not None:
            DF = DF.reindex(columns=self.columns, fill_value=0)
        return DF

    def save(self, path):
        joblib.dump({'medians': self.medians, 'columns': self.columns}, path)

    def load(self, path):
        state = joblib.load(path)
        self.medians = state['medians']
        self.columns = state['columns']

    def _engineer_features(self, df):
        rng = np.random.default_rng(42)
        # ========== NUMERIC DERIVATIVES ==========
        df["LoanToIncomeRatio"] = (df["LoanAmount"] / df["Income"]).replace([np.inf, -np.inf], np.nan)
        df["DebtBurdenIndex"] = df["DTIRatio"] * df["LoanToIncomeRatio"]
        df["EffectiveInterestExposure"] = (df["InterestRate"] * df["LoanTerm"]) / 12
        df["RiskAdjustedLoan"] = df["LoanAmount"] * (1 / df["CreditScore"].replace(0, np.nan))
        df["CreditUtilizationRatio"] = df["LoanAmount"] / (df["NumCreditLines"] * df["Income"] / 12 + 1e-6)
        df["EmploymentStabilityScore"] = np.log1p(df["MonthsEmployed"])
        df["CreditLinesPerYear"] = df["NumCreditLines"] / ((df["MonthsEmployed"] / 12) + 1)
        df["CareerContinuityIndex"] = (df["MonthsEmployed"] / (df["LoanTerm"] + 1)) * 10
        df["IncomeStabilityIndex"] = (1 / (1 + df["DTIRatio"])) * np.log1p(df["MonthsEmployed"])
        df["FinancialHealthIndex"] = (
            (df["CreditScore"] / 850) * 0.5 +
            (1 - df["LoanToIncomeRatio"].clip(0, 2)) * 0.3 +
            (1 - df["DTIRatio"].clip(0, 1.5)) * 0.2
        )
        df["BorrowingPressureScore"] = (
            (df["InterestRate"] / 100) * 0.4 +
            (df["LoanTerm"] / 60) * 0.3 +
            (df["DTIRatio"].clip(0, 2)) * 0.3
        )
        df["CreditMaturityIndex"] = (df["LoanTerm"] / 12) / (df["NumCreditLines"] + 1)
        df["IncomeCreditScoreInteraction"] = df["Income"] * df["CreditScore"]
        df["LoanTermByInterest"] = df["LoanTerm"] * df["InterestRate"]
        df["CombinedDebtIndex"] = df["DTIRatio"] * df["LoanToIncomeRatio"]

        # ========== CATEGORICAL/ORDINAL SIGNALS ==========
        if "Education" in df.columns and df["Education"].dtype == object:
            df["EducationLevelIndex"] = df["Education"].map(self.edu_map).fillna(0).astype(int)
        else:
            df["EducationLevelIndex"] = 0

        def _flag(col, value):
            return (df[col] == value).astype(float) if (col in df.columns and df[col].dtype == object) else 0.0
        is_full_time = _flag("EmploymentType", "Full-time")
        is_self_emp  = _flag("EmploymentType", "Self-employed")
        is_unemp     = _flag("EmploymentType", "Unemployed")
        is_part_time = _flag("EmploymentType", "Part-time")
        is_married   = _flag("MaritalStatus", "Married")
        df["EmploymentStabilityWeight"] = (
            1.0 * is_full_time +
            0.7 * is_part_time +
            0.6 * is_self_emp +
            0.0 * is_unemp
        ).replace(0, np.nan).fillna(0.5)

        # ========== SYNTHETIC / SEMANTIC FEATURES ==========
        # Digital identity & verification
        base_dfp = 0.65 + 0.1 * df["EducationLevelIndex"] / 4 + 0.08 * is_full_time + 0.05 * np.log1p(df["Income"]) / np.log(1e6)
        df["DigitalFootprintScore"] = np.clip(base_dfp + rng.normal(0, 0.08, len(df)), 0, 1)
        # --- Digital identity & verification ---
        # Ensure HasMortgage is numeric 0/1 for this computation:
        if "HasMortgage" in df.columns:
            # Map Yes/No or 1/0 to 1/0 (as int)
            if df["HasMortgage"].dtype == object:
                has_mort = df["HasMortgage"].map({"Yes":1,"No":0}).fillna(0)
            else:
                has_mort = df["HasMortgage"].fillna(0).astype(float)
        else:
            has_mort = 0.0
        
        lam_pvc = 1.5 + 0.05 * (df["Age"] / 10) + 0.4 * has_mort
        # Clip lambda to reasonable range for poisson (avoid too-large/too-small):
        lam_pvc = np.clip(lam_pvc, 0.1, 5)
        df["ProfileVerificationCount"] = np.clip(rng.poisson(lam=lam_pvc), 0, 5).astype(int)

        low = (df["Age"] * 0.3).clip(lower=0)
        high = (df["Age"] * 0.8).clip(lower=0)
        df["IdentityStabilityYears"] = np.clip(rng.uniform(low, high), 0, None).round(1)
        p_multi = np.clip(0.3 + 0.2 * (df["EducationLevelIndex"]/4) + 0.1 * np.log1p(df["Income"])/np.log(1e6), 0, 0.95)
        df["MultiPlatformPresence"] = rng.binomial(1, p_multi).astype(int)
        # Transactional behavior
        mu_opr = 0.7 + 0.1 * (df["Income"] / 100_000) - 0.1 * df["DTIRatio"]
        df["OnlinePurchaseReliability"] = np.clip(mu_opr + rng.normal(0, 0.12, len(df)), 0, 1)
        mu_upc = 0.6 + 0.2 * (np.log1p(df["MonthsEmployed"])/5) + 0.05 * is_full_time
        df["UtilityPaymentConsistency"] = np.clip(mu_upc + rng.normal(0, 0.08, len(df)), 0, 1)
        mu_esr = 0.15 + 0.0005 * (35 - df["Age"])
        df["EcommerceSpendingRatio"] = np.clip(mu_esr + rng.normal(0, 0.04, len(df)), 0, 0.4)
        lam_dsc = 1.5 + 0.5 * (df["EducationLevelIndex"]) + 0.001 * (df["Income"] / 1000)
        df["DigitalSubscriptionCount"] = np.clip(rng.poisson(lam=np.clip(lam_dsc, 0.1, 10)), 0, 10).astype(int)
        # Social trust & sentiment
        mu_sts = 0.5 + 0.05 * (df["EducationLevelIndex"]) + 0.05 * is_full_time
        df["SocialTrustScore"] = np.clip(mu_sts + rng.normal(0, 0.12, len(df)), 0, 1)
        lam_pcc = 2 - 2 * df["SocialTrustScore"]
        df["PublicComplaintCount"] = np.clip(rng.poisson(lam=np.clip(lam_pcc, 0.05, 5)), 0, 5).astype(int)
        lam_end = 1 + 0.5 * is_full_time + 0.5 * (df["EducationLevelIndex"])
        df["EndorsementCount"] = np.clip(rng.poisson(lam=np.clip(lam_end, 0.05, 10)), 0, 10).astype(int)
        df["ReviewSentimentIndex"] = np.clip(df["SocialTrustScore"] + rng.normal(0, 0.1, len(df)), 0, 1)
        # Communication & responsiveness
        mu_lat = 24 - 12 * df["DigitalFootprintScore"] - 6 * is_full_time
        df["ResponseLatencyAvgHrs"] = np.clip(mu_lat + rng.normal(0, 4, len(df)), 1, 48).round(1)
        mu_vrr = 0.75 + 0.1 * is_full_time + 0.05 * (df["EducationLevelIndex"]/4)
        df["VerificationResponseRate"] = np.clip(mu_vrr + rng.normal(0, 0.08, len(df)), 0, 1)
        lam_adc = 1 + 0.002 * (df["Income"] / 1000)
        df["ActiveDeviceCount"] = np.clip(rng.poisson(lam=np.clip(lam_adc, 0.1, 6)), 1, 6).astype(int)
        # Digital financial literacy
        mu_dfl = 0.5 + 0.1 * (df["EducationLevelIndex"]) - 0.003 * df["Age"]
        df["DigitalFinanceLiteracyScore"] = np.clip(mu_dfl + rng.normal(0, 0.08, len(df)), 0, 1)
        mu_mbu = 0.5 - 0.004 * df["Age"] + 0.1 * (np.log1p(df["Income"]) / np.log(1e6))
        df["MobileBankingUsageLevel"] = np.clip(mu_mbu + rng.normal(0, 0.08, len(df)), 0, 1)
        lam_fac = 2 + 0.002 * (df["Income"] / 1000) - 0.03 * (df["Age"] - 30)
        df["FinancialAppCount"] = np.clip(rng.poisson(lam=np.clip(lam_fac, 0.1, 10)), 0, 10).astype(int)
        # Public records & legal standing
        lam_prd = 2 - 1.5 * df["DigitalFootprintScore"]
        df["PublicRecordDiscrepancyCount"] = np.clip(rng.poisson(lam=np.clip(lam_prd, 0.05, 4)), 0, 4).astype(int)
        lam_lic = 3 - 2 * df["SocialTrustScore"]
        df["LegalInquiryCount"] = np.clip(rng.poisson(lam=np.clip(lam_lic, 0.05, 5)), 0, 5).astype(int)
        df["AddressVerificationLevel"] = np.clip(df["DigitalFootprintScore"] + rng.normal(0, 0.08, len(df)), 0, 1)
        # Civic engagement
        mu_cei = 0.3 + 0.1 * (df["EducationLevelIndex"]) + 0.0005 * (df["Age"] + df["Income"] / 10_000)
        df["CivicEngagementIndex"] = np.clip(mu_cei + rng.normal(0, 0.08, len(df)), 0, 1)
        p_vr = np.clip(0.5 + 0.01 * (df["Age"] - 25), 0.0, 0.95)
        df["VoterRegistrationStatus"] = rng.binomial(1, p_vr).astype(int)
        lam_cac = 1 + 0.5 * is_married
        df["CommunityAffiliationCount"] = np.clip(rng.poisson(lam=np.clip(lam_cac, 0.1, 6)), 0, 6).astype(int)
        # Professional & occupational signals
        mu_pes = 0.5 + 0.2 * is_full_time + 0.1 * is_self_emp
        df["ProfessionalEndorsementScore"] = np.clip(mu_pes + rng.normal(0, 0.08, len(df)), 0, 1)
        mu_jcf = 3 - np.log1p(df["MonthsEmployed"]) / 2
        df["JobChangeFrequency5Y"] = np.clip(mu_jcf + rng.normal(0, 0.5, len(df)), 0, 5)
        mu_pwps = 0.4 + 0.2 * is_self_emp + 0.1 * (df["EducationLevelIndex"])
        df["PublicWorkPortfolioScore"] = np.clip(mu_pwps + rng.normal(0, 0.12, len(df)), 0, 1)
        # Aggregated composites
        df["DigitalTrustComposite"] = df[["DigitalFootprintScore", "SocialTrustScore", "AddressVerificationLevel"]].mean(axis=1)
        df["CivicResponsibilityIndex"] = df[["CivicEngagementIndex", "VoterRegistrationStatus"]].mean(axis=1)
        df["OnlineBehaviorIndex"] = df[["OnlinePurchaseReliability", "UtilityPaymentConsistency", "MobileBankingUsageLevel"]].mean(axis=1)
        # Cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

class LoanPreprocessor:
    def __init__(self):
        self.cat_cols = None
        self.num_cols = None
        self.target_col = None
        self.label_encoders = {}
        self.scaler = None

    def fit_transform(self, df):
        dfp = df.copy()
        # Drop identifier columns
        dfp.drop(columns=["LoanID"], inplace=True, errors="ignore")
        # 1. Normalize binary flags
        for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
            if col in dfp.columns:
                if dfp[col].dtype == object:
                    dfp[col] = dfp[col].str.strip().map({"Yes": 1, "No": 0})
                dfp[col] = dfp[col].fillna(0).astype("int8")
        # 2. Identify column groups
        self.cat_cols = [c for c in ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"] if c in dfp.columns]
        self.target_col = "Default" if "Default" in dfp.columns else None
        self.num_cols = [c for c in dfp.columns if c not in self.cat_cols + [self.target_col] and pd.api.types.is_numeric_dtype(dfp[c])]
        # 3. Handle missing values
        for c in self.num_cols:
            dfp[c] = dfp[c].replace([np.inf, -np.inf], np.nan)
            dfp[c] = dfp[c].astype("float64")
            dfp[c] = dfp[c].fillna(dfp[c].median())
        for c in self.cat_cols:
            dfp[c] = dfp[c].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}).fillna("Unknown")
        if self.target_col:
            dfp[self.target_col] = dfp[self.target_col].astype("int8")
        # 4. Label encoding for categoricals
        self.label_encoders = {}
        for c in self.cat_cols:
            le = LabelEncoder()
            dfp[c] = le.fit_transform(dfp[c].astype(str))
            self.label_encoders[c] = le
        # 5. Downcast numerics
        for c in self.num_cols:
            if pd.api.types.is_float_dtype(dfp[c]):
                dfp[c] = pd.to_numeric(dfp[c], downcast="float")
            elif pd.api.types.is_integer_dtype(dfp[c]):
                dfp[c] = pd.to_numeric(dfp[c], downcast="integer")
        # 6. Apply RobustScaler on numeric features
        self.scaler = RobustScaler()
        dfp[self.num_cols] = self.scaler.fit_transform(dfp[self.num_cols])
        # 7. Final cleanup
        dfp = dfp.replace([np.inf, -np.inf], np.nan)
        for c in self.num_cols:
            dfp[c] = dfp[c].fillna(dfp[c].median())
        # 8. Return as DF (final, ready-to-use DataFrame)
        global DF
        DF = dfp
        return DF

    def transform(self, df):
        dfp = df.copy()
        dfp.drop(columns=["LoanID"], inplace=True, errors="ignore")
        # 1. Normalize binary flags
        for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
            if col in dfp.columns:
                if dfp[col].dtype == object:
                    dfp[col] = dfp[col].str.strip().map({"Yes": 1, "No": 0})
                dfp[col] = dfp[col].fillna(0).astype("int8")
        # 2. Handle missing values for numerics/cats (use previously set cols)
        for c in self.num_cols:
            dfp[c] = dfp[c].replace([np.inf, -np.inf], np.nan)
            dfp[c] = dfp[c].astype("float64")
            dfp[c] = dfp[c].fillna(dfp[c].median())
        for c in self.cat_cols:
            dfp[c] = dfp[c].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}).fillna("Unknown")
        if self.target_col and self.target_col in dfp.columns:
            dfp[self.target_col] = dfp[self.target_col].astype("int8")
        # 3. Label encoding (use fitted encoders, handle unseen as 'Unknown')
        for c in self.cat_cols:
            le = self.label_encoders[c]
            vals = dfp[c].astype(str)
            unseen = ~vals.isin(le.classes_)
            if unseen.any():
                vals[unseen] = "Unknown"
                le_classes = np.append(le.classes_, "Unknown")
                le.classes_ = le_classes
            dfp[c] = le.transform(vals)
        # 4. Downcast numerics
        for c in self.num_cols:
            if pd.api.types.is_float_dtype(dfp[c]):
                dfp[c] = pd.to_numeric(dfp[c], downcast="float")
            elif pd.api.types.is_integer_dtype(dfp[c]):
                dfp[c] = pd.to_numeric(dfp[c], downcast="integer")
        # 5. Scale numerics
        dfp[self.num_cols] = self.scaler.transform(dfp[self.num_cols])
        # 6. Final cleanup
        dfp = dfp.replace([np.inf, -np.inf], np.nan)
        for c in self.num_cols:
            dfp[c] = dfp[c].fillna(dfp[c].median())
        global DF
        DF = dfp
        return DF

    def save(self, path):
        state = {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "target_col": self.target_col,
            "label_encoders": self.label_encoders,
            "scaler": self.scaler
        }
        joblib.dump(state, path)

    def load(self, path):
        state = joblib.load(path)
        self.cat_cols = state["cat_cols"]
        self.num_cols = state["num_cols"]
        self.target_col = state["target_col"]
        self.label_encoders = state["label_encoders"]
        self.scaler = state["scaler"]


# ========== LOAD ARTIFACTS AND MODEL ==========
ARTIFACTS_DIR = Path('artifacts')
MODELS_DIR = Path('models')
LGBM_MODEL_PATH = MODELS_DIR / "LGBM_model.pkl"
FEATURE_ENG_PATH = ARTIFACTS_DIR / "feature_engineering_artifacts.pkl"
PREPROCESS_PATH = ARTIFACTS_DIR / "preprocessing_artifacts.pkl"

feature_engineer = LoanFeatureEngineer()
feature_engineer.load(FEATURE_ENG_PATH)
preprocessor = LoanPreprocessor()
preprocessor.load(PREPROCESS_PATH)
model = joblib.load(LGBM_MODEL_PATH)
explainer = shap.TreeExplainer(model)

# ========== FLASK SETUP ==========
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

@app.route("/health", methods=["GET"])
def health() -> object:
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict() -> object:
    try:
        form_data = request.get_json(force=True)
        logger.info("/predict called - received request body")
        logger.debug("request.json=%s", json.dumps(form_data, separators=(",", ":"))[:400])
    except Exception:
        logger.exception("Invalid JSON body")
        return jsonify({"error": "Invalid JSON body"}), 400

    try:
        prediction = predict_lgbm(form_data)
        logger.info("Model output: %s", json.dumps(prediction, separators=(",", ":"))[:400])
        verdict_text = generate_verdict_with_gpt(prediction, form_data)
        prediction["verdict"] = verdict_text
        logger.info("/predict completed successfully")
        return jsonify(prediction)
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Internal error during prediction"}), 500

def predict_lgbm(form_data):
    if isinstance(form_data, dict):
        df_input = pd.DataFrame([form_data])
    else:
        df_input = pd.DataFrame(form_data)
    # 1. Feature engineering
    df_feat = feature_engineer.transform(df_input)
    # 2. Preprocessing (scaling, label encoding)
    df_ready = preprocessor.transform(df_feat)
    if "Default" in df_ready.columns:
        df_ready = df_ready.drop(columns=["Default"])
    # 3. Prediction and SHAP
    prob = float(model.predict_proba(df_ready)[:, 1][0])
    pred = int(model.predict(df_ready)[0])
    shap_values = explainer.shap_values(df_ready)
    shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    feature_names = df_ready.columns.tolist()
    contrib = dict(zip(feature_names, shap_vals))
    abs_importance = dict(zip(feature_names, np.abs(shap_vals)))
    contrib_sorted = {k: v for k, v in sorted(contrib.items(), key=lambda item: -abs(item[1]))}
    abs_importance_sorted = {k: v for k, v in sorted(abs_importance.items(), key=lambda item: -item[1])}
    return {
        "prediction": pred,
        "probability": prob,
        "shap_contribution": contrib_sorted,
        "shap_importance": abs_importance_sorted
    }

def generate_verdict_with_gpt(prediction, form_data):
    api_key = os.environ.get("OPENAI_API_KEY")
    default_probability = float(prediction.get("probability", 0.0))
    risk_level = "HIGH" if default_probability > 0.7 else ("MEDIUM" if default_probability > 0.4 else "LOW")
    top_factors = ", ".join(list(prediction["shap_importance"].keys())[:3])
    shap_json = json.dumps({k: float(v) for k, v in list(prediction['shap_contribution'].items())[:5]}, separators=(",", ":"))
    prompt = (
        f"You are a senior credit risk analyst. Given the model output below, explain to a loan officer in plain English.\n"
        f"Default probability: {default_probability * 100:.1f}%\n"
        f"Risk level: {risk_level}\n"
        f"Top factors: {top_factors}\n"
        f"Applicant: {json.dumps(form_data)[:400]}\n"
        f"SHAP contributions: {shap_json}\n"
        "Recommend whether to approve or decline, with reasons.\n"
        "IMPORTANT: Respond with plain text only. Do not use markdown formatting, code blocks, or any special formatting."
    )
    if not api_key:
        fallback = {
            "HIGH": f"High default risk ({default_probability:.2%}). Main concerns: {top_factors}. Recommend decline or stricter terms.",
            "MEDIUM": f"Moderate default risk ({default_probability:.2%}). Monitor key factors: {top_factors}. Consider standard or slightly stricter terms.",
            "LOW": f"Low default risk ({default_probability:.2%}). Strong applicant due to {top_factors}. Recommend approval.",
        }
        return fallback[risk_level]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert credit risk assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=160,
        )
        content = (response.choices[0].message.content or "").strip()
        return content
    except Exception:
        logger.exception("OpenAI API call failed, using fallback verdict.")
        fallback = {
            "HIGH": f"High default risk ({default_probability:.2%}). Main concerns: {top_factors}. Recommend decline or stricter terms.",
            "MEDIUM": f"Moderate default risk ({default_probability:.2%}). Monitor key factors: {top_factors}. Consider standard or slightly stricter terms.",
            "LOW": f"Low default risk ({default_probability:.2%}). Strong applicant due to {top_factors}. Recommend approval.",
        }
        return fallback[risk_level]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
