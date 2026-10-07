# 💻 Laptop Price Analyzer & Prediction System

An end-to-end Machine Learning and Data Science web application built with **Python, NumPy, Pandas, Scikit-learn, and Streamlit** for laptop price exploratory data analysis, multi-model regression training, real-time price prediction, hardware comparison, and budget-oriented recommendations.

---

## 🎯 1. Project Objective
To analyze laptop pricing patterns, discover relationships between technical specifications (CPU, RAM, GPU, Storage, Display Resolution, Weight) and retail prices in Indian Rupees (₹ INR), and accurately estimate market prices using Machine Learning regression algorithms.

This project is tailored for academic project presentations, viva examinations, and technical demonstrations.

---

## 🚀 2. Key Features

- **🏠 Executive Dashboard**: Key market KPIs (Average Price, Price Range, Catalog Count, Best ML Model, R² score).
- **📊 Exploratory Data Analysis (EDA)**: Interactive Plotly visualizations for Brand price dispersion, RAM scaling, Storage tiers, Processor performance tiers, and overall price distributions.
- **🔮 Real-Time Price Prediction**: Select hardware specs (Brand, CPU, RAM, Storage, GPU, OS, Screen, Weight, Resolution) to compute the model-estimated price and confidence interval.
- **🔎 Laptop Search & Multi-Filter**: Filter the catalog by Brand, RAM, Price Range, and sort by price or value. Supports **CSV Export**.
- **⚖️ Side-by-Side Laptop Comparison**: Compare 2 or 3 laptops with automated difference highlights (lowest price, highest RAM, best storage, highest value score).
- **💰 Budget-Based Recommendation Engine**: Enter any budget (e.g. ₹60,000) to receive ranked laptop recommendations with a deal opportunity tag (*Potentially Good Value* vs *Higher Than Expected*).
- **⭐ Value-for-Money Scoring**: Objective specification-to-price ratio score (0–100) based on component tiers and pricing brackets.
- **🤖 Model Performance & Evaluation**: In-depth comparison of Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor across MAE, MSE, RMSE, and R² Score.
- **📂 Dynamic Dataset Upload**: Upload custom CSV datasets with automatic schema column detection and dynamic retraining.

---

## 🛠️ 3. Technologies Used

- **Python 3.10+**: Core programming language
- **NumPy**: Numerical operations, vector calculations, and mathematical transformations
- **Pandas**: Data loading, cleaning, schema adaptation, missing-value imputation, and multi-criteria querying
- **Scikit-learn**: Preprocessing pipelines (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), train/test split, regression modeling, and metric calculations
- **Streamlit**: Interactive web application framework
- **Plotly & Matplotlib**: Interactive charts and data visualizations
- **Joblib**: Model serialization and persistence (`trained_model.pkl`)

---

## 🧠 4. Machine Learning Workflow

```text
CSV Dataset (data/laptop_prices.csv)
                ↓
Pandas & NumPy Preprocessing (Schema adaptation, unit parsing: RAM "16GB", Storage "512GB")
                ↓
Feature Engineering (Processor Tier, GPU Tier, Display Pixels, Value Score)
                ↓
Train / Test Split (80% Train, 20% Test, random_state=42)
                ↓
Pipeline Construction (StandardScaler for Linear, OneHotEncoder for Categoricals)
                ↓
Model Training (Linear Regression, Random Forest, Gradient Boosting)
                ↓
Evaluation (MAE, MSE, RMSE, R² Score)
                ↓
Best Model Selection & Disk Persistence (models/trained_model.pkl)
                ↓
Real-Time Inference & Deal Opportunity Classification
```

### Regression Algorithms:
1. **Linear Regression**: Baseline parametric model modeling linear relationships.
2. **Random Forest Regressor**: Ensemble of decision trees capturing complex non-linear feature interactions without feature scaling.
3. **Gradient Boosting Regressor**: Sequential boosting algorithm iteratively reducing residual errors.

### Evaluation Metrics:
- **MAE (Mean Absolute Error)**: Average magnitude of prediction errors in ₹.
- **MSE (Mean Squared Error)**: Mean squared deviation emphasizing larger residuals.
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals, directly interpretable in ₹.
- **R² Score (Coefficient of Determination)**: Proportion of variance in laptop prices explained by specifications.

---

## 📂 5. Project Structure

```text
Laptop_Price_Analyzer/
│
├── app.py                      # Main Streamlit web application
│
├── data/
│   └── laptop_prices.csv       # Default real-world laptop dataset (110+ models)
│
├── models/
│   ├── trained_model.pkl       # Serialized best Scikit-learn regression pipeline
│   └── model_metrics.json      # Evaluation metrics (MAE, RMSE, R² score)
│
├── src/
│   ├── data_preprocessing.py   # Dynamic schema mapping, unit cleaner, feature engineering
│   ├── data_analysis.py        # EDA aggregations and Plotly visual figures
│   ├── model_training.py       # Scikit-learn multi-model training and evaluation
│   ├── prediction.py           # Real-time inference and uncertainty margin calculation
│   ├── recommendation.py       # Budget suitability matching and deal evaluation
│   └── comparison.py           # Multi-laptop matrix comparison and highlight extraction
│
├── utils/
│   └── helpers.py              # Currency formatter (₹ INR), validation helpers
│
├── requirements.txt            # Python dependencies
└── README.md                   # Complete academic documentation
```

---

## ⚡ 6. Installation & Execution Guide

### Step 1: Clone or Navigate to the Project Folder
```bash
cd Laptop_Price_Analyzer
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 📊 7. Dataset Information & Customization

The default dataset is located at:
```text
data/laptop_prices.csv
```

### Supported Columns (Dynamic Alias Recognition):
The preprocessor automatically maps and adapts to your dataset columns:
- **Brand**: `Brand`, `Company`, `Manufacturer`
- **Model**: `Model`, `Model_Name`, `Laptop_Name`
- **Processor**: `Processor`, `CPU`, `Processor_Type`
- **RAM**: `RAM`, `RAM_GB`, `Memory` (Handles `"16GB"`, `"16 GB"`, `16`)
- **Storage**: `Storage`, `Storage_GB`, `Disk` (Handles `"512GB"`, `"1TB"`, `1024`)
- **Storage Type**: `Storage_Type`, `Drive_Type` (`SSD`, `HDD`, `eMMC`)
- **GPU**: `GPU`, `Graphics`, `Graphic_Card`
- **Screen Size**: `Screen_Size`, `Inches`, `Display_Size`
- **Operating System**: `Operating_System`, `OS`
- **Weight**: `Weight_kg`, `Weight` (Handles `"1.65 kg"`, `1.65`)
- **Resolution**: `Resolution`, `Display_Resolution` (e.g. `1920x1080`, `2560x1600`)
- **Price**: `Price`, `Price_INR`, `Selling_Price`, `Laptop_Price`

### How to use your own dataset:
1. Replace `data/laptop_prices.csv` with your CSV file, OR
2. Upload it directly through the sidebar in the running web application.

---

## 🧪 8. How to Test Price Prediction

1. Open the application and navigate to **🔮 Price Prediction** in the sidebar.
2. Select your desired regression model (or keep the automatically selected best model).
3. Set your specifications:
   - **Brand**: `Dell`
   - **Processor**: `Intel Core i7 13th Gen`
   - **RAM**: `16 GB`
   - **Storage**: `1024 GB (1 TB)`
   - **GPU**: `NVIDIA RTX 4060`
   - **Screen Size**: `15.6"`
   - **Operating System**: `Windows 11`
4. Click **🚀 Predict Laptop Price**.
5. View the estimated price formatted in ₹ INR along with the statistical confidence interval and component performance tiers.

---

## 🔮 9. Future Enhancements
- Integration with live e-commerce APIs (Amazon, Flipkart) for real-time price tracking.
- Battery life and thermal performance regression analysis.
- Multi-currency conversion for international markets.
