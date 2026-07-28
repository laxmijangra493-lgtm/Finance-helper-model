# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import pdfplumber
# import re
# from dataclasses import dataclass
# from typing import List, Tuple, Optional
# from sklearn.pipeline import FeatureUnion, Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.utils.validation import check_is_fitted

# # ==========================================
# # 1. MACHINE LEARNING BACKEND
# # ==========================================

# @dataclass
# class TxnPrediction:
#     cleaned_text: str
#     predicted_category: str
#     confidence: float
#     needs_review: bool
#     top_k: List[Tuple[str, float]]

# class ProductionTransactionClassifier:
#     CATEGORIES = [
#         "Salary/Income", "Loan EMI", "Investments", "Credit Card Payment",
#         "Subscriptions", "Groceries", "Food & Dining", "Travel & Fuel",
#         "Entertainment", "Shopping", "Bills & Utilities",
#         "Personal Transfer", "Uncategorized"
#     ]

#     def __init__(self, confidence_threshold: float = 0.72, random_state: int = 42):
#         self.confidence_threshold = confidence_threshold
#         self.random_state = random_state
#         self.model: Optional[Pipeline] = None

#     @staticmethod
#     def clean_text(raw_text: str) -> str:
#         text = str(raw_text).upper().strip()
#         text = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", " ", text)
#         text = text.replace("@", " AT ")
#         text = re.sub(r"[/_.\-:|]+", " ", text)
#         banking_noise = (
#             r"\b(?:UPI|IMPS|NEFT|RTGS|POS|ATM|UTR|RRN|TXN|TRANSACTION|"
#             r"REFERENCE|REF|INFO|MOBILE|TRANSFER|PAYMENT ID|ID NO)\b"
#         )
#         text = re.sub(banking_noise, " ", text)
#         text = re.sub(r"\b\d{6,}\b", " ", text)
#         text = re.sub(r"([A-Z]+)\d+([A-Z]*)", r" \1 \2 ", text)
#         text = re.sub(r"\d+([A-Z]+)", r" \1 ", text)
#         text = re.sub(r"[^A-Z\s&]", " ", text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text

#     @staticmethod
#     def merchant_group_key(raw_text: str) -> str:
#         text = ProductionTransactionClassifier.clean_text(raw_text)
#         stop = {
#             "TO", "FROM", "BY", "AT", "ONLINE", "INDIA", "PAY", "PAYMENT",
#             "TRANSFER", "UPI", "IMPS", "BANK", "MOBILE", "SERVICE"
#         }
#         tokens = [t for t in text.split() if t not in stop and len(t) > 2]
#         return " ".join(tokens[:4]) if tokens else "UNKNOWN"

#     @staticmethod
#     def weak_label(raw_text: str) -> str:
#         t = str(raw_text).upper()
#         if re.search(r"\b(?:SALARY|SAL|PAYROLL|PAY\s?ROLL|CMS\s*/?\s*SAL|ACH\s*/?\s*SAL|SAL\s*CREDIT)\b", t): return "Salary/Income"
#         if re.search(r"\b(?:HOME LOAN|LOAN EMI|EMI|ECS EMI|NACH EMI)\b", t): return "Loan EMI"
#         if re.search(r"\b(?:MUTUAL FUND|SIP|FIXED DEPOSIT|FD BOOKING|GROWW|ZERODHA|COIN|KUVERA)\b", t): return "Investments"
#         if re.search(r"\b(?:CREDIT CARD|CARD PAYMENT|CARD MONTHLY DUE|CC PAYMENT|CRED)\b", t): return "Credit Card Payment"
#         if re.search(r"\b(?:NETFLIX|SPOTIFY|YOUTUBE PREMIUM|ICLOUD|APPLE COM BILL|HOTSTAR)\b", t): return "Subscriptions"
#         if re.search(r"\b(?:SWIGGY INSTAMART|BLINKIT|BIGBASKET|KIRANA|GROCERY|DMART|JIOMART)\b", t): return "Groceries"
#         if re.search(r"\b(?:ZOMATO|SWIGGY|DOMINOS|MCDONALDS|BURGER KING|STARBUCKS|CAFE|RESTAURANT)\b", t): return "Food & Dining"
#         if re.search(r"\b(?:UBER|OLA|RAPIDO|PETROL|FUEL|HPCL|IOCL|BPCL|MAKEMYTRIP|IRCTC|AIR INDIA|INDIGO)\b", t): return "Travel & Fuel"
#         if re.search(r"\b(?:PVR|INOX|CINEMA|MOVIE|BOOKMYSHOW)\b", t): return "Entertainment"
#         if re.search(r"\b(?:AMAZON(?!\s*PAY)|MYNTRA|ZARA|DECATHLON|AJIO|FLIPKART|SHOPPING)\b", t): return "Shopping"
#         if re.search(r"\b(?:JIO|AIRTEL|VI |VODAFONE|ELECTRICITY|WATER|GAS|BROADBAND|FASTAG|INSURANCE|LIC)\b", t): return "Bills & Utilities"
#         if re.search(r"\b(?:TRANSFER TO|SENT TO|P2P|PERSONAL TRANSFER|FRIEND|SELF TRANSFER|PAID TO|RECEIVED FROM)\b", t): return "Personal Transfer"
#         return "Uncategorized"

#     @staticmethod
#     def add_features(df: pd.DataFrame) -> pd.DataFrame:
#         out = df.copy()
#         desc = out["Description"].fillna("").astype(str)
#         up = desc.str.upper()

#         out["cleaned_text"] = desc.apply(ProductionTransactionClassifier.clean_text)
#         out["amount"] = pd.to_numeric(out["Amount"], errors="coerce").fillna(0.0)
#         out["abs_amount"] = out["amount"].abs()
#         out["log_amount"] = np.log1p(out["abs_amount"])

#         out["is_credit"] = up.str.contains(r"\b(?:CREDITED|SAL CREDIT|BY CASH DEPOSIT|ACH CR|NEFT CR|RECEIVED FROM)\b", regex=True).astype(int)
#         out["has_emi"] = up.str.contains(r"\b(?:EMI|LOAN|ECS|NACH)\b", regex=True).astype(int)
#         out["has_investment"] = up.str.contains(r"\b(?:SIP|MUTUAL FUND|FIXED DEPOSIT|FD|GROWW|ZERODHA|COIN|KUVERA)\b", regex=True).astype(int)
#         out["has_transfer"] = up.str.contains(r"\b(?:TRANSFER|SENT TO|P2P|SELF TRANSFER|IMPS OUT|IFT OUT|PAID TO)\b", regex=True).astype(int)
#         out["has_upi"] = up.str.contains(r"\bUPI\b|@", regex=True).astype(int)
#         out["has_bill_hint"] = up.str.contains(r"\b(?:ELECTRICITY|WATER|GAS|BROADBAND|RECHARGE|FASTAG|INSURANCE|LIC)\b", regex=True).astype(int)
#         out["has_travel_hint"] = up.str.contains(r"\b(?:UBER|OLA|RAPIDO|PETROL|FUEL|IRCTC|MAKEMYTRIP|INDIGO|AIR)\b", regex=True).astype(int)
#         out["has_food_hint"] = up.str.contains(r"\b(?:ZOMATO|SWIGGY|RESTAURANT|CAFE|EATERY)\b", regex=True).astype(int)
#         out["has_grocery_hint"] = up.str.contains(r"\b(?:KIRANA|GROCERY|INSTAMART|BLINKIT|BIGBASKET|JIOMART)\b", regex=True).astype(int)
#         out["has_cc_hint"] = up.str.contains(r"\b(?:CREDIT CARD|CARD PAYMENT|CC PAYMENT|CRED)\b", regex=True).astype(int)
#         out["has_salary_hint"] = up.str.contains(r"\b(?:SALARY|SAL|PAYROLL|CMS/?SAL|ACH_SAL|ACH SAL)\b", regex=True).astype(int)
#         out["has_amazon_pay"] = up.str.contains(r"\bAMAZON PAY\b|\bAMAZONPAY\b", regex=True).astype(int)
#         out["has_paytm"] = up.str.contains(r"\bPAYTM\b", regex=True).astype(int)
#         out["has_phonepe"] = up.str.contains(r"\bPHONEPE\b", regex=True).astype(int)
#         out["merchant_key"] = desc.apply(ProductionTransactionClassifier.merchant_group_key)

#         return out

#     def build_pipeline(self) -> Pipeline:
#         text_features = FeatureUnion([
#             ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0, sublinear_tf=True, strip_accents="unicode")),
#             ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True))
#         ])

#         numeric_cols = [
#             "amount", "abs_amount", "log_amount", "is_credit", "has_emi",
#             "has_investment", "has_transfer", "has_upi", "has_bill_hint",
#             "has_travel_hint", "has_food_hint", "has_grocery_hint",
#             "has_cc_hint", "has_salary_hint", "has_amazon_pay",
#             "has_paytm", "has_phonepe"
#         ]

#         preprocessor = ColumnTransformer([
#             ("text", text_features, "cleaned_text"),
#             ("merchant", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True), "merchant_key"),
#             ("num", Pipeline([
#                 ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
#                 ("scaler", StandardScaler())
#             ]), numeric_cols)
#         ])

#         return Pipeline([
#             ("preprocessor", preprocessor),
#             ("classifier", LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, class_weight="balanced", random_state=self.random_state))
#         ])

#     def fit(self, df: pd.DataFrame, y: pd.Series):
#         X = self.add_features(df)
#         self.model = self.build_pipeline()
#         self.model.fit(X, y)
#         return self

#     def predict_one(self, description: str, amount: float) -> TxnPrediction:
#         check_is_fitted(self.model)
#         X = pd.DataFrame([{"Description": description, "Amount": amount}])
#         Xf = self.add_features(X)

#         probs = self.model.predict_proba(Xf)[0]
#         classes = self.model.classes_
#         best_idx = int(np.argmax(probs))

#         predicted = classes[best_idx]
#         confidence = float(probs[best_idx])
#         top_idx = np.argsort(probs)[::-1][:3]
#         top_k = [(classes[i], float(probs[i])) for i in top_idx]

#         return TxnPrediction(
#             cleaned_text=Xf.iloc[0]["cleaned_text"],
#             predicted_category=predicted,
#             confidence=confidence,
#             needs_review=confidence < self.confidence_threshold,
#             top_k=top_k
#         )

# # ==========================================
# # 2. PDF PARSER (GOOGLE PAY & STANDARD)
# # ==========================================

# def parse_pdf_statement(uploaded_file) -> pd.DataFrame:
#     records = []
    
#     with pdfplumber.open(uploaded_file) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
            
#             lines = text.split('\n')
            
#             i = 0
#             while i < len(lines):
#                 line = lines[i].strip()
                
#                 # RegEx to detect Date patterns (e.g., "03 Apr, 2026" or "01-06-2026")
#                 date_match = re.search(r'(\d{1,2}\s+[A-Za-z]{3},\s+\d{4}|\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4})', line)
                
#                 if date_match:
#                     date_str = date_match.group(1)
                    
#                     # Look ahead for transaction details & amount in next lines
#                     desc = ""
#                     amount = 0.0
                    
#                     # Search up to next 4 lines for transaction details
#                     for j in range(i, min(i + 4, len(lines))):
#                         sub_line = lines[j]
                        
#                         # Strip out secondary noise lines like UPI ID and Bank Info
#                         if re.search(r'(UPI Transaction ID|Paid by|Paid to Bank|Received in)', sub_line, re.I):
#                             continue
                            
#                         # Extract description keywords
#                         if "Paid to" in sub_line or "Received from" in sub_line:
#                             desc = sub_line.strip()
#                         elif desc == "" and not re.search(r'₹|\bAM\b|\bPM\b', sub_line):
#                             desc += " " + sub_line.strip()
                            
#                         # Extract Amount (e.g., ₹23,581.53 or ₹200)
#                         amt_match = re.search(r'₹\s*([\d,]+(?:\.\d{1,2})?)', sub_line)
#                         if amt_match:
#                             amt_str = amt_match.group(1).replace(',', '')
#                             amount = float(amt_str)
                            
#                     if desc and amount > 0:
#                         records.append({
#                             "Date": date_str,
#                             "Description": desc.strip(),
#                             "Amount": amount
#                         })
#                 i += 1

#     df = pd.DataFrame(records)
#     return df

# # ==========================================
# # 3. STREAMLIT FRONTEND & DASHBOARD
# # ==========================================

# st.set_page_config(
#     page_title="💰 Financial Tracking Dashboard",
#     page_icon="💰",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# st.title("💰 AI Financial Tracking Dashboard")
# st.markdown("Upload your **Google Pay PDF Statement** or **Bank Statement CSV** to auto-categorize transactions and track your finances.")

# uploaded_file = st.file_uploader("Drop your PDF or CSV file here", type=["pdf", "csv"])

# if uploaded_file is not None:
#     file_extension = uploaded_file.name.split('.')[-1].lower()
    
#     with st.spinner("Reading uploaded statement file..."):
#         if file_extension == 'pdf':
#             df = parse_pdf_statement(uploaded_file)
#         else:
#             df = pd.read_csv(uploaded_file)

#     if df.empty or not {'Date', 'Description', 'Amount'}.issubset(df.columns):
#         st.error("Could not parse data properly. Please ensure the file contains valid transactions.")
#     else:
#         # Run ML Categorization On-The-Fly if 'Predicted_Category' doesn't exist
#         if 'Predicted_Category' not in df.columns:
#             with st.spinner("AI is analyzing and predicting categories..."):
#                 classifier = ProductionTransactionClassifier()
#                 y_train = df['Description'].apply(classifier.weak_label)
#                 classifier.fit(df, y_train)
                
#                 predictions = []
#                 for _, row in df.iterrows():
#                     pred = classifier.predict_one(row['Description'], row['Amount'])
#                     predictions.append(pred.predicted_category)
#                 df['Predicted_Category'] = predictions

#         # --- DATA PROCESSING ---
#         df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
#         df = df.dropna(subset=['Date'])
#         df = df.sort_values(by='Date')
        
#         df['Month'] = df['Date'].dt.strftime('%b %Y')
#         df['Month_Key'] = df['Date'].dt.to_period('M') 
        
#         # High level classification
#         def classify_type(row):
#             cat = row['Predicted_Category']
#             desc = str(row['Description']).upper()
            
#             if cat == 'Salary/Income' or "RECEIVED FROM" in desc: 
#                 return 'Income'
#             elif cat == 'Investments': 
#                 return 'Investment'
#             else: 
#                 return 'Expense'
                
#         df['Transaction_Type'] = df.apply(classify_type, axis=1)

#         # Expandable Data Preview Table
#         with st.expander("👀 View Extracted & Categorized Data Preview"):
#             st.dataframe(df[['Date', 'Description', 'Amount', 'Predicted_Category', 'Transaction_Type']], use_container_width=True)

#         # Aggregation Logic
#         monthly_summary = df.groupby(['Month_Key', 'Month', 'Transaction_Type'])['Amount'].sum().reset_index()
#         pivot_summary = monthly_summary.pivot(index=['Month_Key', 'Month'], columns='Transaction_Type', values='Amount').fillna(0).reset_index()
        
#         for col in ['Income', 'Expense', 'Investment']:
#             if col not in pivot_summary.columns:
#                 pivot_summary[col] = 0

#         pivot_summary = pivot_summary.sort_values(by='Month_Key')

#         total_income = df[df['Transaction_Type'] == 'Income']['Amount'].sum()
#         total_expense = df[df['Transaction_Type'] == 'Expense']['Amount'].sum()
#         total_investment = df[df['Transaction_Type'] == 'Investment']['Amount'].sum()

#         # --- DASHBOARD VISUALIZATIONS ---
#         st.markdown("---")
        
#         # 1. TOP METRICS
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Total Income", f"₹{total_income:,.2f}")
#         col2.metric("Total Expenses", f"₹{total_expense:,.2f}")
#         col3.metric("Total Investments", f"₹{total_investment:,.2f}")
            
#         st.markdown("<br>", unsafe_allow_html=True)

#         # 2. CHART 4: Income vs Expenses (Grouped Column Chart)
#         st.subheader("📊 Income vs Expenses (Monthly)")
#         fig_inc_exp = go.Figure()
#         fig_inc_exp.add_trace(go.Bar(
#             x=pivot_summary['Month'], y=pivot_summary['Income'],
#             name='Income', marker_color='#2ECC71',
#             text=pivot_summary['Income'].apply(lambda x: f"₹{x:,.0f}" if x > 0 else ""), textposition='auto'
#         ))
#         fig_inc_exp.add_trace(go.Bar(
#             x=pivot_summary['Month'], y=pivot_summary['Expense'],
#             name='Expenses', marker_color='#E74C3C',
#             text=pivot_summary['Expense'].apply(lambda x: f"₹{x:,.0f}" if x > 0 else ""), textposition='auto'
#         ))
#         fig_inc_exp.update_layout(barmode='group', xaxis_title="Month", yaxis_title="Amount (₹)", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
#         st.plotly_chart(fig_inc_exp, use_container_width=True)

#         st.markdown("<br>", unsafe_allow_html=True)
#         col_left, col_right = st.columns(2)

#         with col_left:
#             # 3. CHART 1: Monthly Expenses Trend
#             st.subheader("📉 Overall Expenses Trend")
#             fig_exp_trend = px.bar(
#                 pivot_summary, x='Month', y='Expense', text_auto='.2s',
#                 labels={'Expense': 'Total Expense (₹)', 'Month': 'Month'},
#                 color_discrete_sequence=['#E74C3C']
#             )
#             fig_exp_trend.update_layout(margin=dict(l=0, r=0, t=30, b=0))
#             st.plotly_chart(fig_exp_trend, use_container_width=True)

#         with col_right:
#             # 4. CHART 3: Expense Breakdown Pie Chart
#             st.subheader("🍕 Where is the money going?")
#             expense_df = df[df['Transaction_Type'] == 'Expense']
#             expense_breakdown = expense_df.groupby('Predicted_Category')['Amount'].sum().reset_index()
            
#             fig_pie = px.pie(
#                 expense_breakdown, values='Amount', names='Predicted_Category', hole=0.4,
#                 hover_data=['Amount'], labels={'Amount': 'Spent (₹)'}
#             )
#             fig_pie.update_traces(textposition='inside', textinfo='percent+label')
#             fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
#             st.plotly_chart(fig_pie, use_container_width=True)

#         st.markdown("<br>", unsafe_allow_html=True)
#         col_bottom_left, col_bottom_right = st.columns(2)
        
#         with col_bottom_left:
#             # 5. CHART 2: Income Tracker
#             st.subheader("📈 Income Tracker")
#             fig_income_line = px.line(
#                 pivot_summary, x='Month', y='Income', markers=True, 
#                 labels={'Income': 'Income (₹)', 'Month': 'Month'}
#             )
#             fig_income_line.update_traces(line=dict(color='#2ECC71', width=3), marker=dict(size=8))
#             fig_income_line.update_layout(margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
#             st.plotly_chart(fig_income_line, use_container_width=True)

#         with col_bottom_right:
#             # 6. CHART 5: Investment Growth
#             st.subheader("💎 Investment Growth")
#             fig_invest = px.bar(
#                 pivot_summary, x='Month', y='Investment', text_auto='.2s',
#                 labels={'Investment': 'Invested (₹)', 'Month': 'Month'},
#                 color_discrete_sequence=['#3498DB']
#             )
#             fig_invest.update_layout(margin=dict(l=0, r=0, t=30, b=0))
#             st.plotly_chart(fig_invest, use_container_width=True)

# else:
#     st.info("👆 Please upload a PDF or CSV bank statement to generate your dashboard.")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

# ==========================================
# 1. MACHINE LEARNING BACKEND
# ==========================================

@dataclass
class TxnPrediction:
    cleaned_text: str
    predicted_category: str
    confidence: float
    needs_review: bool
    top_k: List[Tuple[str, float]]

class ProductionTransactionClassifier:
    CATEGORIES = [
        "Salary/Income", "Loan EMI", "Investments", "Credit Card Payment",
        "Subscriptions", "Groceries", "Food & Dining", "Travel & Fuel",
        "Entertainment", "Shopping", "Bills & Utilities",
        "Personal Transfer", "Uncategorized"
    ]

    def __init__(self, confidence_threshold: float = 0.72, random_state: int = 42):
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.model: Optional[Pipeline] = None

    @staticmethod
    def clean_text(raw_text: str) -> str:
        text = str(raw_text).upper().strip()
        text = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", " ", text)
        text = text.replace("@", " AT ")
        text = re.sub(r"[/_.\-:|]+", " ", text)
        banking_noise = (
            r"\b(?:UPI|IMPS|NEFT|RTGS|POS|ATM|UTR|RRN|TXN|TRANSACTION|"
            r"REFERENCE|REF|INFO|MOBILE|TRANSFER|PAYMENT ID|ID NO)\b"
        )
        text = re.sub(banking_noise, " ", text)
        text = re.sub(r"\b\d{6,}\b", " ", text)
        text = re.sub(r"([A-Z]+)\d+([A-Z]*)", r" \1 \2 ", text)
        text = re.sub(r"\d+([A-Z]+)", r" \1 ", text)
        text = re.sub(r"[^A-Z\s&]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def merchant_group_key(raw_text: str) -> str:
        text = ProductionTransactionClassifier.clean_text(raw_text)
        stop = {
            "TO", "FROM", "BY", "AT", "ONLINE", "INDIA", "PAY", "PAYMENT",
            "TRANSFER", "UPI", "IMPS", "BANK", "MOBILE", "SERVICE"
        }
        tokens = [t for t in text.split() if t not in stop and len(t) > 2]
        return " ".join(tokens[:4]) if tokens else "UNKNOWN"

    @staticmethod
    def weak_label(raw_text: str) -> str:
        t = str(raw_text).upper()
        if re.search(r"\b(?:SALARY|SAL|PAYROLL|PAY\s?ROLL|CMS\s*/?\s*SAL|ACH\s*/?\s*SAL|SAL\s*CREDIT)\b", t): return "Salary/Income"
        if re.search(r"\b(?:HOME LOAN|LOAN EMI|EMI|ECS EMI|NACH EMI)\b", t): return "Loan EMI"
        if re.search(r"\b(?:MUTUAL FUND|SIP|FIXED DEPOSIT|FD BOOKING|GROWW|ZERODHA|COIN|KUVERA)\b", t): return "Investments"
        if re.search(r"\b(?:CREDIT CARD|CARD PAYMENT|CARD MONTHLY DUE|CC PAYMENT|CRED)\b", t): return "Credit Card Payment"
        if re.search(r"\b(?:NETFLIX|SPOTIFY|YOUTUBE PREMIUM|ICLOUD|APPLE COM BILL|HOTSTAR)\b", t): return "Subscriptions"
        if re.search(r"\b(?:SWIGGY INSTAMART|BLINKIT|BIGBASKET|KIRANA|GROCERY|DMART|JIOMART)\b", t): return "Groceries"
        if re.search(r"\b(?:ZOMATO|SWIGGY|DOMINOS|MCDONALDS|BURGER KING|STARBUCKS|CAFE|RESTAURANT)\b", t): return "Food & Dining"
        if re.search(r"\b(?:UBER|OLA|RAPIDO|PETROL|FUEL|HPCL|IOCL|BPCL|MAKEMYTRIP|IRCTC|AIR INDIA|INDIGO)\b", t): return "Travel & Fuel"
        if re.search(r"\b(?:PVR|INOX|CINEMA|MOVIE|BOOKMYSHOW)\b", t): return "Entertainment"
        if re.search(r"\b(?:AMAZON(?!\s*PAY)|MYNTRA|ZARA|DECATHLON|AJIO|FLIPKART|SHOPPING)\b", t): return "Shopping"
        if re.search(r"\b(?:JIO|AIRTEL|VI |VODAFONE|ELECTRICITY|WATER|GAS|BROADBAND|FASTAG|INSURANCE|LIC)\b", t): return "Bills & Utilities"
        if re.search(r"\b(?:TRANSFER TO|SENT TO|P2P|PERSONAL TRANSFER|FRIEND|SELF TRANSFER|PAID TO|RECEIVED FROM)\b", t): return "Personal Transfer"
        return "Uncategorized"

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        desc = out["Description"].fillna("").astype(str)
        up = desc.str.upper()

        out["cleaned_text"] = desc.apply(ProductionTransactionClassifier.clean_text)
        out["amount"] = pd.to_numeric(out["Amount"], errors="coerce").fillna(0.0)
        out["abs_amount"] = out["amount"].abs()
        out["log_amount"] = np.log1p(out["abs_amount"])

        out["is_credit"] = up.str.contains(r"\b(?:CREDITED|SAL CREDIT|BY CASH DEPOSIT|ACH CR|NEFT CR|RECEIVED FROM)\b", regex=True).astype(int)
        out["has_emi"] = up.str.contains(r"\b(?:EMI|LOAN|ECS|NACH)\b", regex=True).astype(int)
        out["has_investment"] = up.str.contains(r"\b(?:SIP|MUTUAL FUND|FIXED DEPOSIT|FD|GROWW|ZERODHA|COIN|KUVERA)\b", regex=True).astype(int)
        out["has_transfer"] = up.str.contains(r"\b(?:TRANSFER|SENT TO|P2P|SELF TRANSFER|IMPS OUT|IFT OUT|PAID TO)\b", regex=True).astype(int)
        out["has_upi"] = up.str.contains(r"\bUPI\b|@", regex=True).astype(int)
        out["has_bill_hint"] = up.str.contains(r"\b(?:ELECTRICITY|WATER|GAS|BROADBAND|RECHARGE|FASTAG|INSURANCE|LIC)\b", regex=True).astype(int)
        out["has_travel_hint"] = up.str.contains(r"\b(?:UBER|OLA|RAPIDO|PETROL|FUEL|IRCTC|MAKEMYTRIP|INDIGO|AIR)\b", regex=True).astype(int)
        out["has_food_hint"] = up.str.contains(r"\b(?:ZOMATO|SWIGGY|RESTAURANT|CAFE|EATERY)\b", regex=True).astype(int)
        out["has_grocery_hint"] = up.str.contains(r"\b(?:KIRANA|GROCERY|INSTAMART|BLINKIT|BIGBASKET|JIOMART)\b", regex=True).astype(int)
        out["has_cc_hint"] = up.str.contains(r"\b(?:CREDIT CARD|CARD PAYMENT|CC PAYMENT|CRED)\b", regex=True).astype(int)
        out["has_salary_hint"] = up.str.contains(r"\b(?:SALARY|SAL|PAYROLL|CMS/?SAL|ACH_SAL|ACH SAL)\b", regex=True).astype(int)
        out["has_amazon_pay"] = up.str.contains(r"\bAMAZON PAY\b|\bAMAZONPAY\b", regex=True).astype(int)
        out["has_paytm"] = up.str.contains(r"\bPAYTM\b", regex=True).astype(int)
        out["has_phonepe"] = up.str.contains(r"\bPHONEPE\b", regex=True).astype(int)
        out["merchant_key"] = desc.apply(ProductionTransactionClassifier.merchant_group_key)

        return out

    def build_pipeline(self) -> Pipeline:
        text_features = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0, sublinear_tf=True, strip_accents="unicode")),
            ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True))
        ])

        numeric_cols = [
            "amount", "abs_amount", "log_amount", "is_credit", "has_emi",
            "has_investment", "has_transfer", "has_upi", "has_bill_hint",
            "has_travel_hint", "has_food_hint", "has_grocery_hint",
            "has_cc_hint", "has_salary_hint", "has_amazon_pay",
            "has_paytm", "has_phonepe"
        ]

        preprocessor = ColumnTransformer([
            ("text", text_features, "cleaned_text"),
            ("merchant", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True), "merchant_key"),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler())
            ]), numeric_cols)
        ])

        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, class_weight="balanced", random_state=self.random_state))
        ])

    def fit(self, df: pd.DataFrame, y: pd.Series):
        X = self.add_features(df)
        self.model = self.build_pipeline()
        self.model.fit(X, y)
        return self

    def predict_one(self, description: str, amount: float) -> TxnPrediction:
        check_is_fitted(self.model)
        X = pd.DataFrame([{"Description": description, "Amount": amount}])
        Xf = self.add_features(X)

        probs = self.model.predict_proba(Xf)[0]
        classes = self.model.classes_
        best_idx = int(np.argmax(probs))

        predicted = classes[best_idx]
        confidence = float(probs[best_idx])
        top_idx = np.argsort(probs)[::-1][:3]
        top_k = [(classes[i], float(probs[i])) for i in top_idx]

        return TxnPrediction(
            cleaned_text=Xf.iloc[0]["cleaned_text"],
            predicted_category=predicted,
            confidence=confidence,
            needs_review=confidence < self.confidence_threshold,
            top_k=top_k
        )

# ==========================================
# 2. PDF PARSER (GOOGLE PAY & STANDARD)
# ==========================================

def parse_pdf_statement(uploaded_file) -> pd.DataFrame:
    records = []
    
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # RegEx to detect Date patterns (e.g., "03 Apr, 2026" or "01-06-2026")
                date_match = re.search(r'(\d{1,2}\s+[A-Za-z]{3},\s+\d{4}|\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4})', line)
                
                if date_match:
                    date_str = date_match.group(1)
                    
                    # Look ahead for transaction details & amount in next lines
                    desc = ""
                    amount = 0.0
                    
                    # Search up to next 4 lines for transaction details
                    for j in range(i, min(i + 4, len(lines))):
                        sub_line = lines[j]
                        
                        # Strip out secondary noise lines like UPI ID and Bank Info
                        if re.search(r'(UPI Transaction ID|Paid by|Paid to Bank|Received in)', sub_line, re.I):
                            continue
                            
                        # Extract description keywords
                        if "Paid to" in sub_line or "Received from" in sub_line:
                            desc = sub_line.strip()
                        elif desc == "" and not re.search(r'₹|\bAM\b|\bPM\b', sub_line):
                            desc += " " + sub_line.strip()
                            
                        # Extract Amount (e.g., ₹23,581.53 or ₹200)
                        amt_match = re.search(r'₹\s*([\d,]+(?:\.\d{1,2})?)', sub_line)
                        if amt_match:
                            amt_str = amt_match.group(1).replace(',', '')
                            amount = float(amt_str)
                            
                    if desc and amount > 0:
                        records.append({
                            "Date": date_str,
                            "Description": desc.strip(),
                            "Amount": amount
                        })
                i += 1

    df = pd.DataFrame(records)
    return df

# ==========================================
# 3. CSV COLUMN AUTOMAPPER
# ==========================================

COLUMN_STACK = {
    'Date': [
        'date', 'txn date', 'transaction date', 'posting date', 'val date', 
        'value date', 'date & time', 'trans date', 'booking date', 'dt'
    ],
    'Description': [
        'description', 'narration', 'particulars', 'transaction details', 
        'remarks', 'details', 'payee', 'transaction remarks', 'memo', 
        'counter party', 'desc', 'transaction description'
    ],
    'Amount': [
        'amount', 'txn amount', 'transaction amount', 'amount (rs.)', 
        'amount(in rs.)', 'net amount', 'total amount', 'val amount', 
        'transaction_amount', 'amt'
    ]
}

def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Matches the user's file columns against our predefined stack."""
    renamed_cols = {}
    
    # Clean the column names from user's file
    user_columns = [str(col).strip() for col in df.columns]
    
    for col in user_columns:
        col_clean = col.lower().strip()
        
        # Check against our stack for Date
        if 'Date' not in renamed_cols.values() and any(alias == col_clean or alias in col_clean for alias in COLUMN_STACK['Date']):
            renamed_cols[col] = 'Date'
            
        # Check against our stack for Description
        elif 'Description' not in renamed_cols.values() and any(alias == col_clean or alias in col_clean for alias in COLUMN_STACK['Description']):
            renamed_cols[col] = 'Description'
            
        # Check against our stack for Amount
        elif 'Amount' not in renamed_cols.values() and any(alias == col_clean or alias in col_clean for alias in COLUMN_STACK['Amount']):
            renamed_cols[col] = 'Amount'

    # Apply automatic renaming
    df = df.rename(columns=renamed_cols)
    
    # Handle banks with separate Debit & Credit columns
    if 'Amount' not in df.columns:
        debit_keywords = ['debit', 'dr', 'withdrawal', 'outflow']
        credit_keywords = ['credit', 'cr', 'deposit', 'inflow']
        
        debit_col = next((c for c in user_columns if any(k in c.lower() for k in debit_keywords)), None)
        credit_col = next((c for c in user_columns if any(k in c.lower() for k in credit_keywords)), None)
        
        if debit_col and credit_col:
            dr = pd.to_numeric(df[debit_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            cr = pd.to_numeric(df[credit_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Amount'] = cr - dr

    return df

# ==========================================
# 4. STREAMLIT FRONTEND & DASHBOARD
# ==========================================

st.set_page_config(
    page_title="💰 Financial Tracking Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("💰 AI Financial Tracking Dashboard")
st.markdown("Upload your **Google Pay PDF Statement** or **Bank Statement CSV** to auto-categorize transactions and track your finances.")

uploaded_file = st.file_uploader("Drop your PDF or CSV file here", type=["pdf", "csv"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("Reading uploaded statement file..."):
        if file_extension == 'pdf':
            df = parse_pdf_statement(uploaded_file)
        else:
            raw_df = pd.read_csv(uploaded_file)
            df = auto_map_columns(raw_df) # Apply intelligent mapping for CSVs

    if df.empty or not {'Date', 'Description', 'Amount'}.issubset(df.columns):
        st.error("⚠️ Could not parse data properly. Please ensure the file contains valid transactions and clear column headers.")
    else:
        # Run ML Categorization On-The-Fly if 'Predicted_Category' doesn't exist
        if 'Predicted_Category' not in df.columns:
            with st.spinner("AI is analyzing and predicting categories..."):
                classifier = ProductionTransactionClassifier()
                y_train = df['Description'].apply(classifier.weak_label)
                classifier.fit(df, y_train)
                
                predictions = []
                for _, row in df.iterrows():
                    pred = classifier.predict_one(row['Description'], row['Amount'])
                    predictions.append(pred.predicted_category)
                df['Predicted_Category'] = predictions

        # --- DATA PROCESSING ---
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values(by='Date')
        
        df['Month'] = df['Date'].dt.strftime('%b %Y')
        df['Month_Key'] = df['Date'].dt.to_period('M') 
        
        # High level classification
        def classify_type(row):
            cat = row['Predicted_Category']
            desc = str(row['Description']).upper()
            
            if cat == 'Salary/Income' or "RECEIVED FROM" in desc: 
                return 'Income'
            elif cat == 'Investments': 
                return 'Investment'
            else: 
                return 'Expense'
                
        df['Transaction_Type'] = df.apply(classify_type, axis=1)

        # Expandable Data Preview Table
        with st.expander("👀 View Extracted & Categorized Data Preview"):
            st.dataframe(df[['Date', 'Description', 'Amount', 'Predicted_Category', 'Transaction_Type']], use_container_width=True)

        # Aggregation Logic
        monthly_summary = df.groupby(['Month_Key', 'Month', 'Transaction_Type'])['Amount'].sum().reset_index()
        pivot_summary = monthly_summary.pivot(index=['Month_Key', 'Month'], columns='Transaction_Type', values='Amount').fillna(0).reset_index()
        
        for col in ['Income', 'Expense', 'Investment']:
            if col not in pivot_summary.columns:
                pivot_summary[col] = 0

        pivot_summary = pivot_summary.sort_values(by='Month_Key')

        total_income = df[df['Transaction_Type'] == 'Income']['Amount'].sum()
        total_expense = df[df['Transaction_Type'] == 'Expense']['Amount'].sum()
        total_investment = df[df['Transaction_Type'] == 'Investment']['Amount'].sum()

        # --- DASHBOARD VISUALIZATIONS ---
        st.markdown("---")
        
        # 1. TOP METRICS
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Income", f"₹{total_income:,.2f}")
        col2.metric("Total Expenses", f"₹{total_expense:,.2f}")
        col3.metric("Total Investments", f"₹{total_investment:,.2f}")
            
        st.markdown("<br>", unsafe_allow_html=True)

        # 2. CHART 4: Income vs Expenses (Grouped Column Chart)
        st.subheader("📊 Income vs Expenses (Monthly)")
        fig_inc_exp = go.Figure()
        fig_inc_exp.add_trace(go.Bar(
            x=pivot_summary['Month'], y=pivot_summary['Income'],
            name='Income', marker_color='#2ECC71',
            text=pivot_summary['Income'].apply(lambda x: f"₹{x:,.0f}" if x > 0 else ""), textposition='auto'
        ))
        fig_inc_exp.add_trace(go.Bar(
            x=pivot_summary['Month'], y=pivot_summary['Expense'],
            name='Expenses', marker_color='#E74C3C',
            text=pivot_summary['Expense'].apply(lambda x: f"₹{x:,.0f}" if x > 0 else ""), textposition='auto'
        ))
        fig_inc_exp.update_layout(barmode='group', xaxis_title="Month", yaxis_title="Amount (₹)", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_inc_exp, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            # 3. CHART 1: Monthly Expenses Trend
            st.subheader("📉 Overall Expenses Trend")
            fig_exp_trend = px.bar(
                pivot_summary, x='Month', y='Expense', text_auto='.2s',
                labels={'Expense': 'Total Expense (₹)', 'Month': 'Month'},
                color_discrete_sequence=['#E74C3C']
            )
            fig_exp_trend.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_exp_trend, use_container_width=True)

        with col_right:
            # 4. CHART 3: Expense Breakdown Pie Chart
            st.subheader("🍕 Where is the money going?")
            expense_df = df[df['Transaction_Type'] == 'Expense']
            expense_breakdown = expense_df.groupby('Predicted_Category')['Amount'].sum().reset_index()
            
            fig_pie = px.pie(
                expense_breakdown, values='Amount', names='Predicted_Category', hole=0.4,
                hover_data=['Amount'], labels={'Amount': 'Spent (₹)'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_bottom_left, col_bottom_right = st.columns(2)
        
        with col_bottom_left:
            # 5. CHART 2: Income Tracker
            st.subheader("📈 Income Tracker")
            fig_income_line = px.line(
                pivot_summary, x='Month', y='Income', markers=True, 
                labels={'Income': 'Income (₹)', 'Month': 'Month'}
            )
            fig_income_line.update_traces(line=dict(color='#2ECC71', width=3), marker=dict(size=8))
            fig_income_line.update_layout(margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
            st.plotly_chart(fig_income_line, use_container_width=True)

        with col_bottom_right:
            # 6. CHART 5: Investment Growth
            st.subheader("💎 Investment Growth")
            fig_invest = px.bar(
                pivot_summary, x='Month', y='Investment', text_auto='.2s',
                labels={'Investment': 'Invested (₹)', 'Month': 'Month'},
                color_discrete_sequence=['#3498DB']
            )
            fig_invest.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_invest, use_container_width=True)

else:
    st.info("👆 Please upload a PDF or CSV bank statement to generate your dashboard.")