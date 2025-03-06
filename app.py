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
CORS(app, resources={r"/*": {"origins": "*"}})

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
    'domains': {
        'business': {
            'columns': {
                'revenue': ['revenue', 'income', 'earnings', 'turnover'],
                'sales': ['sales', 'orders', 'transactions'],
                'profit': ['profit', 'margin', 'net income'],
                'cost': ['cost', 'expense', 'spending'],
                'investment': ['investment', 'funding', 'capital'],
                'market': ['market', 'opportunity', 'demand'],
                'strategy': ['strategy', 'plan', 'business model'],
                'budget': ['budget', 'expenditure', 'allocation']
            }
        },
        'healthcare': {
            'columns': {
                'patient': ['patient', 'client', 'case'],
                'diagnosis': ['diagnosis', 'condition', 'symptom'],
                'treatment': ['treatment', 'therapy', 'intervention'],
                'medical': ['medical', 'health', 'clinical'],
                'hospital': ['hospital', 'clinic', 'facility']
            }
        },
        'education': {
            'columns': {
                'student': ['student', 'pupil', 'learner'],
                'grade': ['grade', 'score', 'mark'],
                'course': ['course', 'class', 'subject', 'curriculum'],
                'teacher': ['teacher', 'instructor', 'professor']
            }
        },
        'technology': {
            'columns': {
                'software': ['software', 'application', 'program', 'app'],
                'hardware': ['hardware', 'device', 'computer', 'equipment'],
                'network': ['network', 'internet', 'connectivity', 'cybersecurity'],
                'data': ['data', 'information', 'database', 'analytics'],
                'AI': ['AI', 'artificial intelligence', 'machine learning', 'neural network']
            }
        },
        'sports': {
            'columns': {
                'game': ['game', 'match', 'competition'],
                'score': ['score', 'result', 'points'],
                'team': ['team', 'squad', 'club'],
                'player': ['player', 'athlete', 'competitor'],
                'tournament': ['tournament', 'league', 'championship']
            }
        },
        'entertainment': {
            'columns': {
                'movie': ['movie', 'film', 'cinema'],
                'music': ['music', 'song', 'album'],
                'tv': ['tv', 'television', 'series'],
                'celebrity': ['celebrity', 'star', 'artist'],
                'show': ['show', 'performance', 'event']
            }
        },
        'politics': {
            'columns': {
                'election': ['election', 'vote', 'ballot'],
                'government': ['government', 'administration', 'executive'],
                'policy': ['policy', 'regulation', 'law'],
                'campaign': ['campaign', 'political', 'candidate'],
                'legislation': ['legislation', 'bill', 'act']
            }
        },
        'science': {
            'columns': {
                'research': ['research', 'study', 'investigation'],
                'experiment': ['experiment', 'trial', 'test'],
                'discovery': ['discovery', 'breakthrough', 'finding'],
                'laboratory': ['laboratory', 'lab', 'facility']
            }
        },
        'finance': {
            'columns': {
                'banking': ['banking', 'bank', 'financial institution'],
                'investment': ['investment', 'invest', 'capital'],
                'stock': ['stock', 'share', 'equity'],
                'portfolio': ['portfolio', 'assets', 'holdings'],
                'trading': ['trading', 'exchange', 'market'],
                'interest': ['interest', 'rate', 'yield'],
                'bond': ['bond', 'debt', 'security']
            }
        },
        'travel': {
            'columns': {
                'flight': ['flight', 'airline', 'aviation'],
                'hotel': ['hotel', 'accommodation', 'lodging'],
                'booking': ['booking', 'reservation', 'schedule'],
                'destination': ['destination', 'location', 'spot'],
                'trip': ['trip', 'journey', 'tour']
            }
        },
        'food': {
            'columns': {
                'restaurant': ['restaurant', 'diner', 'eatery'],
                'cuisine': ['cuisine', 'food', 'dish'],
                'recipe': ['recipe', 'cooking', 'preparation'],
                'meal': ['meal', 'dinner', 'lunch', 'breakfast'],
                'chef': ['chef', 'cook', 'culinary']
            }
        },
        'lifestyle': {
            'columns': {
                'fashion': ['fashion', 'style', 'clothing'],
                'beauty': ['beauty', 'cosmetic', 'makeup'],
                'wellness': ['wellness', 'health', 'fitness'],
                'leisure': ['leisure', 'hobby', 'recreation'],
                'home': ['home', 'house', 'residence'],
                'design': ['design', 'interior', 'architecture']
            }
        },
        'environment': {
            'columns': {
                'climate': ['climate', 'weather', 'temperature'],
                'sustainability': ['sustainability', 'renewable', 'eco-friendly'],
                'ecology': ['ecology', 'ecosystem', 'biodiversity'],
                'conservation': ['conservation', 'protection', 'preservation'],
                'pollution': ['pollution', 'contamination', 'waste']
            }
        },
        'automotive': {
            'columns': {
                'car': ['car', 'automobile', 'vehicle'],
                'engine': ['engine', 'motor', 'powertrain'],
                'driving': ['driving', 'road', 'transportation'],
                'fuel': ['fuel', 'gasoline', 'diesel'],
                'tire': ['tire', 'rubber', 'wheel']
            }
        },
        'real_estate': {
            'columns': {
                'property': ['property', 'real estate', 'asset'],
                'housing': ['housing', 'home', 'residence'],
                'mortgage': ['mortgage', 'loan', 'financing'],
                'rent': ['rent', 'rental', 'lease'],
                'sale': ['sale', 'sell', 'market'],
                'listing': ['listing', 'advertisement', 'catalog'],
                'realtor': ['realtor', 'agent', 'broker']
            }
        },
        'retail': {
            'columns': {
                'store': ['store', 'shop', 'boutique'],
                'shopping': ['shopping', 'consumer', 'purchase'],
                'product': ['product', 'item', 'merchandise'],
                'inventory': ['inventory', 'stock', 'supply'],
                'brand': ['brand', 'label', 'trademark']
            }
        },
        'telecommunications': {
            'columns': {
                'network': ['network', 'telecom', 'communication'],
                'mobile': ['mobile', 'cellular', 'smartphone'],
                'broadband': ['broadband', 'internet', 'connectivity'],
                'signal': ['signal', 'coverage', 'reception'],
                'wireless': ['wireless', 'wifi', 'bluetooth']
            }
        },
        'agriculture': {
            'columns': {
                'farming': ['farming', 'agriculture', 'cultivation'],
                'crop': ['crop', 'harvest', 'produce'],
                'livestock': ['livestock', 'animal', 'farm'],
                'irrigation': ['irrigation', 'watering', 'drip'],
                'soil': ['soil', 'earth', 'land'],
                'organic': ['organic', 'natural', 'bio']
            }
        },
        'energy': {
            'columns': {
                'oil': ['oil', 'petroleum', 'crude'],
                'gas': ['gas', 'natural gas', 'fuel'],
                'electricity': ['electricity', 'power', 'energy'],
                'renewable': ['renewable', 'solar', 'wind', 'hydro'],
                'nuclear': ['nuclear', 'atomic', 'radiation']
            }
        },
        'legal': {
            'columns': {
                'law': ['law', 'legal', 'regulation'],
                'litigation': ['litigation', 'lawsuit', 'court'],
                'attorney': ['attorney', 'lawyer', 'counsel'],
                'contract': ['contract', 'agreement', 'deal'],
                'compliance': ['compliance', 'regulation', 'standards'],
                'justice': ['justice', 'court', 'verdict']
            }
        },
        'media': {
            'columns': {
                'alertname': ['alertname', 'alertstate', 'report','alert','alertname'],
                'severity': ['severity','severity','service',],
                'job': ['job'],
                'first_triggered': ['activeAt', 'first_triggered', 'last_triggered','severity','description']
            }
        },
        
        'others': {
            'columns': {
                'unknown': []  # If no matching keywords are found, this category is used.
            }
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
        # Convert date columns to datetime
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass
        
        # Handle missing values
        self.df_cleaned = self.df.copy()
        for col in self.df_cleaned.columns:
            if self.df_cleaned[col].dtype in ['int64', 'float64']:
                self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)
            else:
                self.df_cleaned[col].fillna('Unknown', inplace=True)

    def generate_visualizations(self) -> List[Dict]:
        """Generate visualizations with smart chart type selection for each data type"""
        visualizations = []
        
        # Function to convert plotly figure to base64
        def plotly_to_base64(fig):
            buf = io.BytesIO()
            fig.write_image(buf, format='png', engine='kaleido')
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Get column types
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        date_cols = self.df.select_dtypes(include=['datetime']).columns
        
        # 1. Correlation Heatmap (only if multiple numeric columns exist)
        if len(numeric_cols) >= 2:
            fig = px.imshow(self.df[numeric_cols].corr(), text_auto=True, color_continuous_scale='Viridis')
            fig.update_layout(title='Correlation Heatmap')
            visualizations.append({
                'title': 'Correlation Analysis',
                'image': plotly_to_base64(fig),
                'type': 'heatmap'
            })
        
        # 2. Smart distribution of numeric columns
        for col in numeric_cols:
            # Check for distribution characteristics
            skewness = self.df[col].skew()
            uniqueness = self.df[col].nunique() / len(self.df[col])
            
            if uniqueness < 0.05:  # Few unique values relative to data size
                # Use bar chart for discrete numeric data
                fig = px.bar(self.df[col].value_counts().sort_index(), 
                            title=f'Distribution of {col}')
            else:
                # Use histogram for continuous data
                fig = px.histogram(self.df, x=col, nbins=50, 
                                title=f'Distribution of {col}')
            
            visualizations.append({
                'title': f'Distribution Analysis - {col}',
                'image': plotly_to_base64(fig),
                'type': 'distribution' if uniqueness >= 0.05 else 'bar',
                'xAxis': col,
                'yAxis': 'Frequency'
            })
        
        # 3. Categorical data visualization
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            unique_count = len(value_counts)
            
            if unique_count <= 5:
                # Use pie chart for few categories
                fig = px.pie(self.df, names=col, title=f'Distribution of {col}')
                chart_type = 'pie'
            elif unique_count <= 10:
                # Use donut chart for moderate number of categories
                fig = px.pie(self.df, names=col, hole=0.3, title=f'Distribution of {col}')
                chart_type = 'donut'
            elif unique_count <= 20:
                # Use bar chart for many categories
                fig = px.bar(value_counts, title=f'Distribution of {col}')
                chart_type = 'bar'
            else:
                # Skip visualization for too many categories
                continue
                
            visualizations.append({
                'title': f'{chart_type.capitalize()} Chart - {col}',
                'image': plotly_to_base64(fig),
                'type': chart_type
            })
        
        # 4. Time series visualization
        if date_cols.size > 0:
            date_col = date_cols[0]  # Use the first date column
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                # Aggregate data by date to avoid overcrowding
                daily_data = self.df.groupby(date_col)[num_col].mean().reset_index()
                fig = px.line(daily_data, x=date_col, y=num_col, 
                            title=f'{num_col} Trend Over Time')
                visualizations.append({
                    'title': f'Time Series - {num_col}',
                    'image': plotly_to_base64(fig),
                    'type': 'line',
                    'xAxis': date_col,
                    'yAxis': num_col
                })
        
        # 5. Relationships between numeric columns (if multiple exist)
        if len(numeric_cols) >= 3:
            # Use bubble chart for three numeric columns
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
        
        # 6. Categorical vs Numerical (choose most interesting relationships)
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            # Select categorical column with moderate number of categories
            for cat_col in categorical_cols:
                if 3 <= self.df[cat_col].nunique() <= 10:
                    # Find numeric column with highest variance
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
                    break  # Only create one such visualization
        
        return visualizations
        

    def generate_insights(self) -> List[str]:
        """Generate text insights for the frontend"""
        insights = []
        
        # Basic dataset insights
        insights.append(f"Dataset contains {len(self.df)} rows and {len(self.df.columns)} columns")
        
        # Missing values analysis
        missing = self.df.isnull().sum()
        if missing.any():
            insights.append(f"Found {missing.sum()} missing values across {sum(missing > 0)} columns")
        
        # Numerical insights
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            stats = self.df[col].describe()
            insights.append(f"{col} summary: Average = {stats['mean']:.2f}, Min = {stats['min']:.2f}, Max = {stats['max']:.2f}")
            
            # Distribution analysis
            skew = self.df[col].skew()
            if abs(skew) > 1:
                insights.append(f"{col} shows {'positive' if skew > 0 else 'negative'} skew ({skew:.2f})")
        
        # Correlation insights
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

def infer_domain(columns):
    """Infer the domain based on column names and check required columns"""
    domain_scores = {}
    required_column_matches = {}

    for domain, info in knowledge_base['domains'].items():
        keywords = set(info.get('keywords', []))
        required_columns = info.get('required_columns', {})

        # Count keyword matches
        keyword_match_count = sum(
            1 for col in columns if any(re.search(r'\b' + re.escape(keyword) + r'\b', col, re.IGNORECASE) 
            for keyword in keywords)
        )
        domain_scores[domain] = keyword_match_count

        # Count required column matches correctly
        matched_required_columns = 0
        for required_col, aliases in required_columns.items():
            # Check if the exact required column name is present
            if required_col in columns:
                matched_required_columns += 1
            # Check if any alias matches
            elif any(alias in columns for alias in aliases):
                matched_required_columns += 1  

        required_column_matches[domain] = matched_required_columns

    # Identify the best-matching domain
    best_match = max(domain_scores.items(), key=lambda x: x[1])[0]

    # Check if at least 50% of the required columns exist
    required_col_count = len(knowledge_base['domains'][best_match].get('required_columns', {}))

    if required_col_count > 0:
        coverage = required_column_matches[best_match] / required_col_count
        print(f"Domain: {best_match}, Required Columns: {required_col_count}, Matched: {required_column_matches[best_match]}, Coverage: {coverage}")  # Debug print
        insights_button_enabled = coverage >= 0.5
    else:
        insights_button_enabled = False  # If no required columns are defined, disable button

    return best_match, domain_scores, insights_button_enabled


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
        
        domain, domain_scores, insights_button_enabled = infer_domain(columns)
        
        return jsonify({
            'columns': columns,
            'detected_domain': domain,
            'domain_scores': domain_scores,
            'all_domains': list(knowledge_base['domains'].keys()),
            'insights_button_enabled': insights_button_enabled,
            'data': df.to_dict('records')
        })
    
    except Exception as e:
        logger.exception("Error processing file upload")
        return jsonify({'error': str(e)}), 500

@app.route('/insights', methods=['POST'])
def generate_insights():
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        df = pd.DataFrame(data.get('data'))
        if df.empty:
            return jsonify({'error': 'Dataset is empty'}), 400
        
        analyzer = DataAnalyzer(df)
        
        with ThreadPoolExecutor() as executor:
            insights_future = executor.submit(analyzer.generate_insights)
            visualizations_future = executor.submit(analyzer.generate_visualizations)
            
            results = {
                'insights': insights_future.result(),
                'visualizations': visualizations_future.result()
            }
        
        return jsonify(results)
        
    except Exception as e:
        logger.exception("Error generating insights")
        return jsonify({'error': str(e)}), 500
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
