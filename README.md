# LendSmart AI Credit Analyzer

A full-stack loan approval prediction application using machine learning models with an interactive Next.js frontend and Flask backend.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm or pnpm

### 1. Backend Setup (Python Flask API)

First, set up and run the backend server:

```bash
# Navigate to backend directory
cd LoanAPP/backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional required packages
pip install flask flask-cors

# Run the Flask server
python pipeline.py
```

The backend server will start on `http://localhost:8000`

**Important**: The backend must be running before starting the frontend, as it serves the ML models and prediction API.

### 2. Frontend Setup (Next.js)

In a new terminal window:

```bash
# Navigate to the main LoanAPP directory
cd LoanAPP

# Install dependencies
npm install
# or if you prefer pnpm
pnpm install

# Start the development server
npm run dev
# or
pnpm dev
```

The frontend will be available at `http://localhost:3000`

## 🔧 Environment Variables

### Backend (.env in LoanAPP/backend/)

Create or update the `.env` file in the backend directory:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME="CrediLens AI Credit Analyzer"
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
PORT=8000
```

### Frontend Environment Variables

The frontend reads environment variables from the backend's `.env` file. Key variables:

- `NEXT_PUBLIC_BACKEND_URL`: Backend API URL (default: http://localhost:8000)
- `NEXT_PUBLIC_APP_NAME`: Application display name
- `OPENAI_API_KEY`: Required for AI-powered verdict generation
- `PORT`: Backend server port (default: 8000)

## 📁 Project Structure

```
LoanAPP/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── backend/               # Python Flask backend
│   ├── api.py            # API test client
│   ├── pipeline.py       # Main Flask app with ML pipeline
│   ├── requirements.txt  # Python dependencies
│   ├── .env             # Environment variables
│   ├── artifacts/       # ML model artifacts
│   ├── models/          # Trained models
│   └── venv/            # Virtual environment
├── components/           # React components
│   ├── charts/          # Chart components
│   ├── ui/              # UI components
│   └── *.tsx            # Feature components
├── hooks/               # Custom React hooks
├── lib/                 # Utility libraries
└── public/              # Static assets
```

## 🤖 Machine Learning Pipeline

The application uses several ML models:

1. **LightGBM**: Primary prediction model
2. **XGBoost**: Alternative model
3. **SHAP**: For feature importance and explainability
4. **OpenAI GPT**: For generating human-readable verdicts

### Model Artifacts

Pre-trained models and preprocessing artifacts are stored in:

- `LoanAPP/backend/artifacts/`
- `LoanAPP/backend/models/`

## 🔗 API Endpoints

### Backend API (Flask)

- `GET /health` - Health check endpoint
- `POST /predict` - Loan prediction endpoint

### Sample Request

```json
{
  "Age": 35,
  "Income": 90000,
  "LoanAmount": 25000,
  "CreditScore": 710,
  "MonthsEmployed": 60,
  "NumCreditLines": 4,
  "InterestRate": 7.2,
  "LoanTerm": 48,
  "DTIRatio": 0.25,
  "Education": "Master's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": 1,
  "HasDependents": 0,
  "LoanPurpose": "Car",
  "HasCoSigner": 0
}
```

## 🛠 Development

### Backend Development

```bash
cd LoanAPP/backend
source venv/bin/activate
python pipeline.py
```

### Frontend Development

```bash
cd LoanAPP
npm run dev
```

### Testing the API

Use the provided test client:

```bash
cd LoanAPP/backend
python api.py
```

## 📦 Dependencies

### Backend (Python)

- numpy, pandas - Data manipulation
- scikit-learn - ML preprocessing
- lightgbm, xgboost - ML models
- shap - Model explainability
- flask, flask-cors - Web framework
- joblib - Model serialization

### Frontend (Next.js)

- React 19 - UI framework
- Next.js 15 - React framework
- Tailwind CSS - Styling
- Radix UI - Component library
- Recharts - Data visualization
- React Hook Form - Form handling
- Zod - Schema validation

## 🚨 Troubleshooting

### Backend Issues

1. **Port already in use**: Change the PORT in `.env` file
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model artifacts missing**: Ensure artifacts are in the correct directories

### Frontend Issues

1. **Backend connection failed**: Ensure backend is running on port 8000
2. **Environment variables**: Check `.env` file in backend directory
3. **Dependencies**: Run `npm install` or `pnpm install`

### Common Issues

- **CORS errors**: Backend includes CORS middleware for cross-origin requests
- **OpenAI API**: Ensure valid API key is set for verdict generation
- **Model loading**: Check that model artifacts exist in backend/artifacts/

## 📄 License

This project is licensed under the terms specified in the LICENSE file.
