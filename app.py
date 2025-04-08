from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yaml
import re
import base64
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from scipy import stats
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
matplotlib.use('Agg')
plt.style.use('ggplot')

# Initialize Flask app
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


knowledge_base_path = r'C:\Users\harsha vardhan\analyser\knowledge_base.yml'

try:
    with open(knowledge_base_path, 'r') as file:
        knowledge_base = yaml.safe_load(file)
except FileNotFoundError:

    knowledge_base = {
    "domains": {
        "ecommerce": {
            "keywords": [
                "order_id", "product_name", "price", "quantity", "customer_id", "category", "sales",
                "discount", "order_date", "city", "payment_method"
            ],
            "relationships": [
                ["order_id", "product_name"],
                ["price", "discount"],
                ["category", "sales"]
            ],
            "required_columns": {
                "order_id": ["orderid", "order_number", "order"],
                "order_date": ["date", "orderdate", "purchase_date"],
                "product_name": ["product", "item_name", "product_id"],
                "price": ["unit_price", "product_price", "amount"],
                "quantity": ["qty", "item_quantity", "units"],
                "customer_id": ["custid", "cust_id", "customerid"],
                "city": ["location", "customer_city", "ship_city"],
                "payment_method": ["payment", "payment_type", "pay_method"]
            }
        },
        "education": {
            "keywords": ["student_id", "grade", "subject", "teacher", "marks", "attendance", "course", "university"],
            "relationships": [
                ["student_id", "marks"],
                ["course", "teacher"],
                ["university", "attendance"]
            ]
        },
        "finance": {
            "keywords": ["revenue", "profit", "expenses", "tax", "cost", "income", "investment", "assets", "liabilities"],
            "relationships": [
                ["revenue", "expenses"],
                ["profit", "tax"],
                ["assets", "liabilities"]
            ]
        },
        "health": {
            "keywords": ["patient_id", "diagnosis", "medication", "hospital_name", "age", "gender", "disease", "doctor_name"],
            "relationships": [
                ["patient_id", "diagnosis"],
                ["patient_id", "medication"],
                ["hospital_name", "doctor_name"]
            ],
            "required_columns": {
                "patient_id": ["pat_id", "patientid"],
                "age": ["patient_age", "years_old"],
                "gender": ["sex", "patient_gender"],
                "diagnosis": ["medical_condition", "disease"],
                "doctor_name": ["physician", "consultant"],
                "hospital_name": ["clinic", "medical_center"]
            }
        },
        "hr": {
            "keywords": ["employee_id", "department", "salary", "designation", "experience", "attendance", "performance", "hiring_date", "termination_date"],
            "relationships": [
                ["employee_id", "department"],
                ["employee_id", "performance"],
                ["designation", "salary"],
                ["hiring_date", "termination_date"]
            ]
        },
        "retail": {
            "keywords": ["store_id", "product", "sales", "inventory", "region", "price", "category", "profit"],
            "relationships": [
                ["store_id", "sales"],
                ["inventory", "product"],
                ["region", "profit"]
            ]
        },
        "transportation": {
            "keywords": ["vehicle_id", "route", "distance", "fuel", "driver", "passenger", "station", "fare"],
            "relationships": [
                ["vehicle_id", "route"],
                ["distance", "fuel"],
                ["station", "fare"]
            ]
        },
        "others": {
            "keywords": ["alertname", "alertstate", "service", "site", "state", "severity", "first_triggered", "last_triggered"],
            "relationships": [
                ["alertname", "alertstate"],
                ["service", "site"],
                ["state", "severity"],
                ["first_triggered", "last_triggered"]
            ]
        }
    }
}
    

    # Define home route
@app.route("/")
def home():
    return jsonify({"message": "Backend is running!"})


class DataAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.sample_size = min(len(df), 10000)
        self.df_sample = df.sample(n=self.sample_size) if len(df) > self.sample_size else df
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the dataframe for analysis"""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass

        self.df_cleaned = self.df.copy()
        for col in self.df_cleaned.columns:
            if self.df_cleaned[col].dtype in ['int64', 'float64']:
                self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)
            else:
                self.df_cleaned[col].fillna('Unknown', inplace=True)

    def generate_visualizations(self) -> List[Dict]:
        visualizations = []

        def plotly_to_base64(fig):
            try:
                buf = io.BytesIO()
                fig.write_image(buf, format='png', engine='kaleido')
                buf.seek(0)
                return base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                logger.error(f"Error converting plotly to base64: {str(e)}")
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="Visualization unavailable", showarrow=False)
                buf = io.BytesIO()
                empty_fig.write_image(buf, format='png', engine='kaleido')
                buf.seek(0)
                return base64.b64encode(buf.getvalue()).decode('utf-8')

        try:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            date_cols = self.df.select_dtypes(include=['datetime']).columns

            if len(numeric_cols) == 0 and len(categorical_cols) == 0 and len(date_cols) == 0:
                fig = go.Figure()
                fig.add_annotation(text="No suitable columns found for visualization", showarrow=False)
                fig.update_layout(title='Dataset Visualization Issue')
                visualizations.append({
                    'title': 'Dataset Analysis',
                    'image': plotly_to_base64(fig),
                    'type': 'info'
                })
                return visualizations

            if len(numeric_cols) >= 2:
                fig = px.imshow(self.df[numeric_cols].corr(), text_auto=True, color_continuous_scale='Viridis')
                fig.update_layout(title='Correlation Heatmap')
                visualizations.append({
                    'title': 'Correlation Analysis',
                    'image': plotly_to_base64(fig),
                    'type': 'heatmap'
                })

            for col in numeric_cols:
                skewness = self.df[col].skew()
                uniqueness = self.df[col].nunique() / len(self.df[col])
                if uniqueness < 0.05:
                    fig = px.bar(self.df[col].value_counts().sort_index(), title=f'Distribution of {col}')
                else:
                    fig = px.histogram(self.df, x=col, nbins=50, title=f'Distribution of {col}')
                visualizations.append({
                    'title': f'Distribution Analysis - {col}',
                    'image': plotly_to_base64(fig),
                    'type': 'distribution' if uniqueness >= 0.05 else 'bar',
                    'xAxis': col,
                    'yAxis': 'Frequency'
                })

            for col in categorical_cols:
                value_counts = self.df[col].value_counts()
                unique_count = len(value_counts)
                if unique_count <= 5:
                    fig = px.pie(self.df, names=col, title=f'Distribution of {col}')
                    chart_type = 'pie'
                elif unique_count <= 10:
                    fig = px.pie(self.df, names=col, hole=0.3, title=f'Distribution of {col}')
                    chart_type = 'donut'
                elif unique_count <= 20:
                    fig = px.bar(value_counts, title=f'Distribution of {col}')
                    chart_type = 'bar'
                else:
                    continue

                visualizations.append({
                    'title': f'{chart_type.capitalize()} Chart - {col}',
                    'image': plotly_to_base64(fig),
                    'type': chart_type
                })

            if date_cols.size > 0:
                date_col = date_cols[0]
                for num_col in numeric_cols[:3]:
                    daily_data = self.df.groupby(date_col)[num_col].mean().reset_index()
                    fig = px.line(daily_data, x=date_col, y=num_col, title=f'{num_col} Trend Over Time')
                    visualizations.append({
                        'title': f'Time Series - {num_col}',
                        'image': plotly_to_base64(fig),
                        'type': 'line',
                        'xAxis': date_col,
                        'yAxis': num_col
                    })

            if len(numeric_cols) >= 3:
                fig = px.scatter(self.df, x=numeric_cols[0], y=numeric_cols[1],
                                 size=numeric_cols[2],
                                 title=f'Relationship: {numeric_cols[0]} vs {numeric_cols[1]} vs {numeric_cols[2]}')
                visualizations.append({
                    'title': 'Multi-variable Relationship',
                    'image': plotly_to_base64(fig),
                    'type': 'bubble',
                    'xAxis': numeric_cols[0],
                    'yAxis': numeric_cols[1]
                })

            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                for cat_col in categorical_cols:
                    if 3 <= self.df[cat_col].nunique() <= 10:
                        variances = self.df[numeric_cols].var()
                        num_col = variances.idxmax()
                        fig = px.box(self.df, x=cat_col, y=num_col,
                                     title=f'{num_col} Distribution by {cat_col}')
                        visualizations.append({
                            'title': f'Box Plot - {num_col} by {cat_col}',
                            'image': plotly_to_base64(fig),
                            'type': 'box',
                            'xAxis': cat_col,
                            'yAxis': num_col
                        })
                        break

        except Exception as e:
            logger.exception(f"Error generating visualizations: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error generating visualizations: {str(e)}", showarrow=False)
            fig.update_layout(title='Visualization Error')
            visualizations.append({
                'title': 'Error in Analysis',
                'image': plotly_to_base64(fig),
                'type': 'error'
            })

        return visualizations

    def generate_insights(self) -> List[str]:
        insights = []
        insights.append(f"Dataset contains {len(self.df)} rows and {len(self.df.columns)} columns")

        missing = self.df.isnull().sum()
        if missing.any():
            insights.append(f"Found {missing.sum()} missing values across {sum(missing > 0)} columns")

        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            stats = self.df[col].describe()
            insights.append(f"{col} summary: Average = {stats['mean']:.2f}, Min = {stats['min']:.2f}, Max = {stats['max']:.2f}")
            skew = self.df[col].skew()
            if abs(skew) > 1:
                insights.append(f"{col} shows {'positive' if skew > 0 else 'negative'} skew ({skew:.2f})")

        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            strong_corrs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        strong_corrs.append(f"{numeric_cols[i]} and {numeric_cols[j]} (correlation: {corr_matrix.iloc[i, j]:.2f})")
            if strong_corrs:
                insights.append("Strong correlations found between:")
                for corr in strong_corrs:
                    insights.append(f"- {corr}")
        return insights

    @staticmethod
    def infer_domain(columns):
        logger.info(f"Starting domain inference with columns: {columns}")
        domain_scores = {}
        for domain, info in knowledge_base['domains'].items():
            keywords = set(info.get('keywords', []))
            keyword_match_count = sum(
                1 for col in columns if any(
                    re.search(r'\b' + re.escape(keyword) + r'\b', col, re.IGNORECASE)
                    for keyword in keywords
                )
            )
            domain_scores[domain] = keyword_match_count
            logger.debug(f"Domain '{domain}' has {keyword_match_count} keyword matches")
        best_match, max_score = max(domain_scores.items(), key=lambda x: x[1])
        if max_score == 0:
            best_match = 'others'
        logger.info(f"Final domain determination: '{best_match}'")
        return best_match, domain_scores


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        domain, domain_scores = DataAnalyzer.infer_domain(columns)
        if all(score == 0 for score in domain_scores.values()):
            domain = 'others'
        return jsonify({
            'columns': columns,
            'detected_domain': domain,
            'domain_scores': domain_scores,
            'all_domains': list(knowledge_base['domains'].keys()),
            'data': df.to_dict('records')
        })
    except Exception as e:
        logger.exception("Error processing file upload")
        return jsonify({'error': str(e)}), 500


@app.route('/insights', methods=['POST'])
def generate_insights_route():
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400

        selected_domain = data.get('domain', 'unknown')
        logger.info(f"Processing insights for domain: {selected_domain}")

        df = pd.DataFrame(data.get('data'))
        if df.empty:
            return jsonify({
                'insights': ["Dataset is empty - no insights available"],
                'visualizations': []
            })

        logger.info(f"Dataset columns: {df.columns.tolist()}")
        logger.info(f"Dataset dtypes: {df.dtypes.to_dict()}")

        analyzer = DataAnalyzer(df)

        insights = analyzer.generate_insights()
        visualizations = analyzer.generate_visualizations()

        return jsonify({
            'insights': insights,
            'visualizations': visualizations
        })

    except Exception as e:
        logger.exception(f"Error in insights endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'insights': ["An error occurred during analysis"],
            'visualizations': []
        })
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
