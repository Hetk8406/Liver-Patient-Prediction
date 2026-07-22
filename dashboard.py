import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import webbrowser
import pickle
from threading import Timer

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Initialize Dash application with bootstrap stylesheet
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Liver Patient Prediction Dashboard"

# Color Palette Config
COLORS = {
    'bg': '#f5f7fa',
    'card': 'white',
    'primary': '#2E86DE',
    'success': '#28A745',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'text': '#2C3E50',
    'borders': '#D6DBDF'
}

# ------------------------------------------------------------------
# DATA LOADING & MODEL TRAINING WORKFLOW
# ------------------------------------------------------------------
DATASET_PATH = 'Data/Indian Liver Patient Dataset (ILPD).csv'

# Default values for UI in case files/data loading fails
data_loaded = False
error_message = None
df = None
df_encoded = None
X_train_scaled = None
X_test_scaled = None
y_train = None
y_test = None
scaler = None
le = None
best_model = None
best_model_name = "N/A"
best_accuracy = 0.0
df_comparison = pd.DataFrame()
trained_models = {}
predictions = {}
probabilities = {}

# Global raw dataframe pointer for stats page
raw_df = None
duplicate_count = 0

try:
    if os.path.exists(DATASET_PATH):
        columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                   'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 
                   'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']
        raw_df = pd.read_csv(DATASET_PATH, names=columns)
        df = raw_df.copy()
        
        # 1. Fill missing values with median
        df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median())
        
        # 2. Drop duplicates
        duplicate_count = df.duplicated().sum()
        df = df.drop_duplicates()
        
        # 3. Label Encoding for Gender
        le = LabelEncoder()
        df_encoded = df.copy()
        df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])
        
        # 4. Separate features and target (Dataset values: 1 = liver disease, 2 = healthy)
        X = df_encoded.drop('Dataset', axis=1)
        y = df_encoded['Dataset'].map({1: 1, 2: 0}) # map to 1 and 0 for binary classification
        
        # 5. Train-Test Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 6. Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifiers
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Support Vector Machine': SVC(probability=True, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        metrics = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            predictions[name] = y_pred
            probabilities[name] = y_prob
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            metrics.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'ROC-AUC': auc
            })
            
        df_comparison = pd.DataFrame(metrics).sort_values(by='F1 Score', ascending=False)
        
        # Load saved files if available from notebook execution
        saved_files_exist = os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('encoder.pkl')
        if saved_files_exist:
            try:
                with open('best_model.pkl', 'rb') as f:
                    best_model = pickle.load(f)
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                with open('encoder.pkl', 'rb') as f:
                    le = pickle.load(f)
                
                # Evaluate accuracy dynamically on test set
                X_test_scaled_saved = scaler.transform(X_test)
                y_pred_saved = best_model.predict(X_test_scaled_saved)
                best_accuracy = accuracy_score(y_test, y_pred_saved)
                
                # Map name
                cname = type(best_model).__name__
                name_mapping = {
                    'RandomForestClassifier': 'Random Forest',
                    'GradientBoostingClassifier': 'Gradient Boosting',
                    'LogisticRegression': 'Logistic Regression',
                    'KNeighborsClassifier': 'K-Nearest Neighbors',
                    'DecisionTreeClassifier': 'Decision Tree',
                    'SVC': 'Support Vector Machine'
                }
                best_model_name = name_mapping.get(cname, cname)
                print("Successfully loaded saved model, scaler, and encoder from disk.")
            except Exception as load_err:
                print("Failed to load saved files, falling back to dynamic best:", load_err)
                saved_files_exist = False
                
        if not saved_files_exist:
            best_model_name = df_comparison.iloc[0]['Model']
            best_accuracy = df_comparison.iloc[0]['Accuracy']
            best_model = trained_models[best_model_name]
        
        data_loaded = True
    else:
        error_message = f"Dataset file not found at: {DATASET_PATH}. Please make sure the data is located correctly."
except Exception as e:
    error_message = f"Error during initialization or model training: {str(e)}"

# ------------------------------------------------------------------
# APP LAYOUT HELPER FUNCTIONS
# ------------------------------------------------------------------

def make_header():
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Liver Patient Prediction Dashboard", className="text-start m-0", style={"color": COLORS['text'], "fontWeight": "bold"}),
                            html.P("Machine Learning Analysis and Disease Prediction", className="text-start text-muted m-0", style={"fontSize": "0.95rem"})
                        ],
                        md=8
                    ),
                    dbc.Col(
                        [
                            html.P(f"System Date/Time: {current_time_str}", className="text-end text-muted m-0", style={"fontSize": "0.9rem", "paddingTop": "10px"})
                        ],
                        md=4
                    )
                ],
                align="center",
                className="py-3 px-4 border-bottom",
                style={"backgroundColor": "white"}
            )
        ]
    )

def make_kpi_cards():
    if not data_loaded:
        return html.Div()
    
    total_records = len(df)
    liver_patients = len(df[df['Dataset'] == 1])
    healthy_patients = len(df[df['Dataset'] == 2])
    
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Total Records", className="card-subtitle text-muted mb-1"),
                            html.H3(f"{total_records}", className="card-title m-0", style={"color": COLORS['primary'], "fontWeight": "bold"})
                        ]
                    ),
                    style={"border": f"1px solid {COLORS['borders']}", "boxShadow": "0 2px 4px rgba(0,0,0,0.02)", "borderRadius": "6px"}
                ),
                md=3
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Liver Disease Patients", className="card-subtitle text-muted mb-1"),
                            html.H3(f"{liver_patients}", className="card-title m-0", style={"color": COLORS['danger'], "fontWeight": "bold"})
                        ]
                    ),
                    style={"border": f"1px solid {COLORS['borders']}", "boxShadow": "0 2px 4px rgba(0,0,0,0.02)", "borderRadius": "6px"}
                ),
                md=3
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Healthy Patients", className="card-subtitle text-muted mb-1"),
                            html.H3(f"{healthy_patients}", className="card-title m-0", style={"color": COLORS['success'], "fontWeight": "bold"})
                        ]
                    ),
                    style={"border": f"1px solid {COLORS['borders']}", "boxShadow": "0 2px 4px rgba(0,0,0,0.02)", "borderRadius": "6px"}
                ),
                md=3
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Prediction Accuracy", className="card-subtitle text-muted mb-1"),
                            html.H3(f"{best_accuracy*100:.2f}%", className="card-title m-0", style={"color": COLORS['warning'], "fontWeight": "bold"}),
                            html.Small(f"({best_model_name})", className="text-muted", style={"fontSize": "0.75rem"})
                        ]
                    ),
                    style={"border": f"1px solid {COLORS['borders']}", "boxShadow": "0 2px 4px rgba(0,0,0,0.02)", "borderRadius": "6px"}
                ),
                md=3
            ),
        ],
        className="my-4 px-3"
    )

def make_sidebar():
    return html.Div(
        [
            html.H5("Navigation", className="px-3 mb-3", style={"color": COLORS['text'], "fontWeight": "bold"}),
            dbc.Nav(
                [
                    dbc.NavLink("Overview", href="/overview", active="exact", className="py-2 px-3 mb-1", style={"borderRadius": "4px"}),
                    dbc.NavLink("Dataset Analysis", href="/dataset-analysis", active="exact", className="py-2 px-3 mb-1", style={"borderRadius": "4px"}),
                    dbc.NavLink("Feature Relationships", href="/feature-relationships", active="exact", className="py-2 px-3 mb-1", style={"borderRadius": "4px"}),
                    dbc.NavLink("Model Performance", href="/model-performance", active="exact", className="py-2 px-3 mb-1", style={"borderRadius": "4px"}),
                    dbc.NavLink("Prediction Tool", href="/prediction-tool", active="exact", className="py-2 px-3 mb-1", style={"borderRadius": "4px"}),
                    dbc.NavLink("About Project", href="/about-project", active="exact", className="py-2 px-3 mb-1", style={"borderRadius": "4px"}),
                ],
                vertical=True,
                pills=True,
                className="px-2"
            )
        ],
        style={
            "backgroundColor": "white",
            "borderRight": f"1px solid {COLORS['borders']}",
            "height": "100%",
            "minHeight": "calc(100vh - 75px)",
            "paddingTop": "20px"
        }
    )

def make_footer():
    return html.Footer(
        [
            html.Div(
                [
                    html.P("Liver Patient Prediction Dashboard", className="m-0 font-weight-bold", style={"color": COLORS['text']}),
                    html.P("Built using Python, Dash, Plotly, Pandas and Scikit-learn", className="m-0 text-muted", style={"fontSize": "0.85rem"})
                ],
                className="text-center py-3 border-top mt-5",
                style={"backgroundColor": "white"}
            )
        ]
    )

# ------------------------------------------------------------------
# APP LAYOUT MAIN FRAME
# ------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        make_header(),
        dbc.Container(
            [
                # Error layout if dataset/model fails to load
                html.Div(
                    id="error-banner",
                    children=[
                        dbc.Alert(
                            error_message,
                            color="danger",
                            dismissable=True,
                            className="mt-3"
                        ) if error_message else html.Div()
                    ]
                ),
                make_kpi_cards(),
                dbc.Row(
                    [
                        dbc.Col(make_sidebar(), md=3, lg=2, style={"padding": "0"}),
                        dbc.Col(
                            html.Div(id="page-content", className="p-4", style={"minHeight": "60vh"}),
                            md=9,
                            lg=10
                        )
                    ]
                )
            ],
            fluid=True,
            style={"backgroundColor": COLORS['bg']}
        ),
        make_footer()
    ],
    style={"backgroundColor": COLORS['bg']}
)

# ------------------------------------------------------------------
# PAGE RENDERERS
# ------------------------------------------------------------------

def render_overview_page():
    if not data_loaded:
        return html.Div("Data not loaded.")
    
    # 1. Gender distribution Pie Chart
    gender_counts = df['Gender'].value_counts()
    gender_pie = px.pie(
        names=gender_counts.index,
        values=gender_counts.values,
        color_discrete_sequence=['#3498DB', '#E74C3C'],
        title="Gender Distribution"
    )
    gender_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
    gender_pie.layout.template = 'plotly_white'
    
    # 2. Target distribution Countplot
    target_counts = df['Dataset'].value_counts()
    target_map = {1: 'Liver Disease (Class 1)', 2: 'Healthy (Class 2)'}
    target_names = [target_map[x] for x in target_counts.index]
    target_bar = px.bar(
        x=target_names,
        y=target_counts.values,
        color=target_names,
        color_discrete_sequence=[COLORS['danger'], COLORS['success']],
        labels={'x': 'Class', 'y': 'Count'},
        title="Target Distribution"
    )
    target_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=280)
    target_bar.layout.template = 'plotly_white'
    
    # 3. Age Distribution Histogram
    age_hist = px.histogram(
        df,
        x='Age',
        nbins=20,
        color_discrete_sequence=['#2E86DE'],
        title="Age Distribution"
    )
    age_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280, yaxis_title="Count")
    age_hist.layout.template = 'plotly_white'
    
    return html.Div(
        [
            html.H4("Project Overview", className="mb-4", style={"color": COLORS['text'], "fontWeight": "bold"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Project Summary & Problem Statement", className="card-title", style={"color": COLORS['text'], "fontWeight": "semibold"}),
                                    html.P(
                                        "Liver disease is a major healthcare challenge, often linked to lifestyle factors, alcohol consumption, "
                                        "and hepatitis. Because early diagnosis significantly increases the survival rate, building predictive models "
                                        "using standard biochemical tests can support medical professionals in identifying high-risk individuals.",
                                        className="card-text text-secondary"
                                    ),
                                    html.P(
                                        "This dashboard presents a machine learning-based classification framework using the Indian Liver Patient Dataset (ILPD). "
                                        "The objective is to train a model that can accurately predict whether a patient has liver disease based on demographic data "
                                        "and biochemical measurements like bilirubin levels, proteins, and liver enzymes.",
                                        className="card-text text-secondary"
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=7
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Dataset Profile Summary", className="card-title", style={"color": COLORS['text'], "fontWeight": "semibold"}),
                                    html.Table(
                                        [
                                            html.Tr([html.Td(html.Strong("Total Records:")), html.Td(f" {len(df)}")], style={"height": "35px"}),
                                            html.Tr([html.Td(html.Strong("Total Columns:")), html.Td(" 11")], style={"height": "35px"}),
                                            html.Tr([html.Td(html.Strong("Target Variable:")), html.Td(" Dataset (1: Patient, 2: Healthy)")], style={"height": "35px"}),
                                            html.Tr([html.Td(html.Strong("Numerical Features:")), html.Td(" 9 Features")], style={"height": "35px"}),
                                            html.Tr([html.Td(html.Strong("Categorical Features:")), html.Td(" Gender (Male/Female)")], style={"height": "35px"})
                                        ],
                                        style={"width": "100%", "marginTop": "10px"}
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=5
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([dcc.Graph(figure=gender_pie, config={'displayModeBar': False})]),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                        ),
                        md=4
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([dcc.Graph(figure=target_bar, config={'displayModeBar': False})]),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                        ),
                        md=4
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([dcc.Graph(figure=age_hist, config={'displayModeBar': False})]),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                        ),
                        md=4
                    )
                ]
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6("Key Insights:", className="card-title text-dark", style={"fontWeight": "bold"}),
                        html.Ul(
                            [
                                html.Li("The dataset suffers from a noticeable gender imbalance, with male subjects outnumbering females."),
                                html.Li("There is a target imbalance where roughly 71.4% of the patients belong to the liver disease category."),
                                html.Li("Most patients are middle-aged adults, with the bulk concentrated in the 30–60 years bracket.")
                            ],
                            className="text-secondary m-0"
                        )
                    ]
                ),
                style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginTop": "20px"}
            )
        ]
    )

def render_dataset_analysis_page():
    if not data_loaded:
        return html.Div("Data not loaded.")
    
    # 1. Missing Values DataFrame computed dynamically
    missing_data = pd.DataFrame({
        'Feature': raw_df.columns,
        'Missing Values': [raw_df[c].isnull().sum() for c in raw_df.columns],
        'Imputation Strategy': ['None' if raw_df[c].isnull().sum() == 0 else f'Imputed with Median ({raw_df[c].median():.2f})' for c in raw_df.columns]
    })
    
    # 2. Summary stats
    summary_stats = df.describe().reset_index().round(2)
    
    # 3. Correlation Heatmap
    corr_matrix = df.drop('Gender', axis=1).corr().round(2)
    corr_heatmap = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',

            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate="%{text}",
            hoverongaps=False
        )
    )
    corr_heatmap.update_layout(title="Feature Correlation Heatmap", height=450, margin=dict(l=50, r=50, t=50, b=50))
    
    return html.Div(
        [
            html.H4("Dataset Analysis & Statistics", className="mb-4", style={"color": COLORS['text'], "fontWeight": "bold"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Feature Descriptions & Missing Values", className="card-title mb-3", style={"fontWeight": "semibold"}),
                                    dash_table.DataTable(
                                        data=missing_data.to_dict('records'),
                                        columns=[{'name': i, 'id': i} for i in missing_data.columns],
                                        style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '13px'},
                                        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                                        style_table={'overflowX': 'auto'}
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=12
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Summary Statistics", className="card-title mb-3", style={"fontWeight": "semibold"}),
                                    dash_table.DataTable(
                                        data=summary_stats.to_dict('records'),
                                        columns=[{'name': i, 'id': i} for i in summary_stats.columns],
                                        style_cell={'textAlign': 'left', 'padding': '6px', 'fontSize': '12px'},
                                        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                                        style_table={'overflowX': 'auto'}
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=12
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([dcc.Graph(figure=corr_heatmap, config={'displayModeBar': False})]),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=8
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Data Quality Notes", className="card-title", style={"fontWeight": "semibold"}),
                                    html.Hr(),
                                    html.Ul(
                                        children=(
                                            [html.Li(f"{c} had {raw_df[c].isnull().sum()} missing rows filled dynamically with the median ({raw_df[c].median():.2f}).") 
                                             for c in raw_df.columns if raw_df[c].isnull().sum() > 0] or [html.Li("No missing values detected in raw data.")]
                                        ) + [
                                            html.Li(f"{duplicate_count} duplicate rows were successfully detected and removed during cleaning.")
                                        ] + [
                                            html.Li(f"{col1} and {col2} are highly correlated ({df.drop('Gender', axis=1).corr().loc[col1, col2]:.2f}).")
                                            for col1 in df.drop('Gender', axis=1).corr().columns
                                            for col2 in df.drop('Gender', axis=1).corr().columns
                                            if col1 != col2 and col1 < col2 and abs(df.drop('Gender', axis=1).corr().loc[col1, col2]) > 0.75
                                        ],
                                        className="text-secondary"
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=4
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Dataset Preview (First 5 Rows)", className="card-title mb-3", style={"fontWeight": "semibold"}),
                                    dash_table.DataTable(
                                        data=df.head(5).to_dict('records'),
                                        columns=[{'name': i, 'id': i} for i in df.columns],
                                        style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '13px'},
                                        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                                        style_table={'overflowX': 'auto'}
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=12
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Feature Distribution and Boxplots (Matching Notebook)", className="card-title mb-3", style={"fontWeight": "semibold"}),
                                    html.Label("Select Laboratory Feature to View Distribution & Outliers:"),
                                    dcc.Dropdown(
                                        id='analysis-feature-dropdown',
                                        options=[{'label': c, 'value': c} for c in df.columns if c not in ['Gender', 'Dataset']],
                                        value='Total_Bilirubin',
                                        clearable=False,
                                        className="mb-4"
                                    ),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id='analysis-dist-plot'), md=6),
                                        dbc.Col(dcc.Graph(id='analysis-box-plot'), md=6)
                                    ])
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                        ),
                        md=12
                    )
                ]
            )
        ]
    )

def render_feature_relationships_page():
    if not data_loaded:
        return html.Div("Data not loaded.")
    
    numerical_features = [c for c in df.columns if c not in ['Gender', 'Dataset']]
    
    return html.Div(
        [
            html.H4("Feature Relationships", className="mb-4", style={"color": COLORS['text'], "fontWeight": "bold"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Control Filters", className="card-title mb-3", style={"fontWeight": "semibold"}),
                                    
                                    html.Label("Choose X-Axis Feature:"),
                                    dcc.Dropdown(
                                        id='x-feature-dropdown',
                                        options=[{'label': f, 'value': f} for f in numerical_features],
                                        value='Age',
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Choose Y-Axis Feature:"),
                                    dcc.Dropdown(
                                        id='y-feature-dropdown',
                                        options=[{'label': f, 'value': f} for f in numerical_features],
                                        value='Total_Bilirubin',
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Filter by Gender:"),
                                    dcc.Dropdown(
                                        id='gender-filter-dropdown',
                                        options=[
                                            {'label': 'All', 'value': 'All'},
                                            {'label': 'Male', 'value': 'Male'},
                                            {'label': 'Female', 'value': 'Female'}
                                        ],
                                        value='All',
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Filter by Target (Dataset):"),
                                    dcc.Dropdown(
                                        id='target-filter-dropdown',
                                        options=[
                                            {'label': 'All', 'value': 'All'},
                                            {'label': 'Liver Patients (Class 1)', 'value': 1},
                                            {'label': 'Healthy Patients (Class 2)', 'value': 2}
                                        ],
                                        value='All',
                                        clearable=False,
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                        ),
                        md=4
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Graph(id='relationship-scatter-plot')
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                        ),
                        md=8
                    )
                ]
            )
        ]
    )

def render_model_performance_page():
    if not data_loaded:
        return html.Div("Data not loaded.")
    
    # Model comparison table formatting
    comp_data = df_comparison.copy()
    for c in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
        comp_data[c] = comp_data[c].apply(lambda val: f"{val*100:.2f}%")
    
    # 1. ROC Curves Figure
    roc_fig = go.Figure()
    models_to_plot = [
        ('Logistic Regression', probabilities['Logistic Regression']),
        ('K-Nearest Neighbors', probabilities['K-Nearest Neighbors']),
        ('Decision Tree', probabilities['Decision Tree']),
        ('Random Forest', probabilities['Random Forest']),
        ('Support Vector Machine', probabilities['Support Vector Machine']),
        ('Gradient Boosting', probabilities['Gradient Boosting'])
    ]
    for name, prob in models_to_plot:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = roc_auc_score(y_test, prob)
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {auc_val:.2f})"))
    
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='black'), name='Random Guess'))
    roc_fig.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    roc_fig.layout.template = 'plotly_white'
    
    # 2. Confusion Matrix of best model
    best_pred = predictions[best_model_name]
    cm = confusion_matrix(y_test, best_pred)
    cm_heatmap = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=['Predicted Healthy (0)', 'Predicted Patient (1)'],
            y=['True Healthy (0)', 'True Patient (1)'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            hoverongaps=False
        )
    )
    cm_heatmap.update_layout(title=f"Confusion Matrix (" + best_model_name + ")", height=380, margin=dict(l=50, r=50, t=50, b=50))
    
    # 3. Feature Importance (Random Forest)
    rf_feat_importance = pd.Series(trained_models['Random Forest'].feature_importances_, index=X_train.columns).sort_values(ascending=True)
    rf_importance_fig = px.bar(
        x=rf_feat_importance.values,
        y=rf_feat_importance.index,
        orientation='h',
        color_discrete_sequence=[COLORS['primary']],
        labels={'x': 'Importance Score', 'y': 'Feature'},
        title="Random Forest Feature Importance"
    )
    rf_importance_fig.update_layout(margin=dict(l=50, r=20, t=40, b=30), height=380)
    rf_importance_fig.layout.template = 'plotly_white'
    
    return html.Div(
        [
            html.H4("Model Performance & Evaluation", className="mb-4", style={"color": COLORS['text'], "fontWeight": "bold"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Classifier Comparison Table", className="card-title mb-3", style={"fontWeight": "semibold"}),
                                    dash_table.DataTable(
                                        data=comp_data.to_dict('records'),
                                        columns=[{'name': i, 'id': i} for i in comp_data.columns],
                                        style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '13px'},
                                        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                                        style_table={'overflowX': 'auto'},
                                        style_data_conditional=[
                                            {
                                                'if': {'filter_query': f'{{Model}} = "{best_model_name}"'},
                                                'backgroundColor': '#D5F5E3',
                                                'fontWeight': 'bold'
                                            }
                                        ]
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=12
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([dcc.Graph(figure=roc_fig, config={'displayModeBar': False})]),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=6
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([dcc.Graph(figure=cm_heatmap, config={'displayModeBar': False})]),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=6
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Graph(figure=rf_importance_fig, config={'displayModeBar': False}),
                                    html.Div(
                                        [
                                            html.P("The model relies most on these biochemical measurements while demographic variables contribute comparatively less.", className="text-muted mt-2 text-center", style={"fontSize": "0.9rem"})
                                        ]
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                        ),
                        md=12
                    )
                ]
            )
        ]
    )

def render_prediction_tool_page():
    return html.Div(
        [
            html.H4("Patient Diagnosis Prediction Tool", className="mb-4", style={"color": COLORS['text'], "fontWeight": "bold"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Input Patient Demographics & Lab Reports", className="card-title mb-4", style={"fontWeight": "semibold"}),
                                    
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Age (Years):"),
                                                    dbc.Input(id="input-age", type="number", value=45, min=1, max=120, className="mb-3")
                                                ],
                                                md=6
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Gender:"),
                                                    dcc.Dropdown(
                                                        id="input-gender",
                                                        options=[
                                                            {'label': 'Male', 'value': 'Male'},
                                                            {'label': 'Female', 'value': 'Female'}
                                                        ],
                                                        value='Male',
                                                        clearable=False,
                                                        className="mb-3"
                                                    )
                                                ],
                                                md=6
                                            )
                                        ]
                                    ),
                                    
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Total Bilirubin:"),
                                                    dbc.Input(id="input-tb", type="number", value=1.2, min=0.1, max=100.0, step=0.1, className="mb-3")
                                                ],
                                                md=6
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Direct Bilirubin:"),
                                                    dbc.Input(id="input-db", type="number", value=0.4, min=0.1, max=50.0, step=0.1, className="mb-3")
                                                ],
                                                md=6
                                            )
                                        ]
                                    ),
                                    
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Alkaline Phosphotase:"),
                                                    dbc.Input(id="input-ap", type="number", value=220, min=10, max=3000, step=1, className="mb-3")
                                                ],
                                                md=4
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Alamine Aminotransferase (SGPT):"),
                                                    dbc.Input(id="input-sgpt", type="number", value=45, min=5, max=3000, step=1, className="mb-3")
                                                ],
                                                md=4
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Aspartate Aminotransferase (SGOT):"),
                                                    dbc.Input(id="input-sgot", type="number", value=50, min=5, max=5000, step=1, className="mb-3")
                                                ],
                                                md=4
                                            )
                                        ]
                                    ),
                                    
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Total Proteins:"),
                                                    dbc.Input(id="input-tp", type="number", value=6.5, min=1.0, max=15.0, step=0.1, className="mb-3")
                                                ],
                                                md=4
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Albumin:"),
                                                    dbc.Input(id="input-alb", type="number", value=3.2, min=0.5, max=10.0, step=0.1, className="mb-3")
                                                ],
                                                md=4
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Albumin and Globulin Ratio:"),
                                                    dbc.Input(id="input-ag", type="number", value=0.9, min=0.1, max=5.0, step=0.1, className="mb-3")
                                                ],
                                                md=4
                                            )
                                        ]
                                    ),
                                    
                                    dbc.Button("Predict Patient Status", id="btn-predict", color="primary", className="mt-3 w-100", style={"fontWeight": "bold"})
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                        ),
                        md=8
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Prediction Result", className="card-title mb-3", style={"fontWeight": "semibold"}),
                                        html.Div(id="prediction-result-display", children=[
                                            dbc.Alert("Enter patient details and click Predict to see the diagnosis results here.", color="info")
                                        ])
                                    ]
                                ),
                                style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "height": "100%", "minHeight": "300px"}
                            )
                        ],
                        md=4
                    )
                ]
            )
        ]
    )

def render_about_page():
    return html.Div(
        [
            html.H4("About the Project", className="mb-4", style={"color": COLORS['text'], "fontWeight": "bold"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Project Objective", style={"fontWeight": "semibold"}),
                                    html.P(
                                        "The primary objective of this project is to develop and analyze predictive machine learning models that can evaluate clinical "
                                        "blood tests and categorize patients as having liver disease or not. Early clinical indicators can help initiate "
                                        "timely medical intervention.",
                                        className="text-secondary"
                                    ),
                                    html.H5("Workflow Implemented", className="mt-4", style={"fontWeight": "semibold"}),
                                    html.Ul(
                                        [
                                            html.Li([html.Strong("Data Cleaning: "), "Filled missing albumin/globulin ratio rows with columns' median value and removed 13 duplicate entries."]),
                                            html.Li([html.Strong("EDA: "), "Visualized dataset feature distributions, target imbalance, and mapped correlation collinearities."]),
                                            html.Li([html.Strong("Feature Preprocessing: "), "Encoded categorical variables, separated target, and split into train/test using stratified partition."]),
                                            html.Li([html.Strong("Model Building: "), "Trained six classifiers: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and Gradient Boosting."]),
                                            html.Li([html.Strong("Model Evaluation: "), "Analyzed models using Accuracy, Precision, Recall, F1 score, and ROC curves."])
                                        ],
                                        className="text-secondary"
                                    ),
                                    html.H5("Challenges Faced", className="mt-4", style={"fontWeight": "semibold"}),
                                    html.Ul(
                                        [
                                            html.Li("Handling class imbalance where liver patients are represented significantly more than healthy individuals."),
                                            html.Li("Managing extreme outliers in liver enzymes values without sacrificing clinical accuracy.")
                                        ],
                                        className="text-secondary"
                                    )
                                ]
                            ),
                            style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                        ),
                        md=8
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Future Improvements", style={"fontWeight": "semibold"}),
                                        html.Ul(
                                            [
                                                html.Li("Collect more dataset observations representing a balanced class partition."),
                                                html.Li("Incorporate hyperparameters tuning using GridSearchCV."),
                                                html.Li("Deploy prediction interface as an API endpoint.")
                                            ],
                                            className="text-secondary"
                                        )
                                    ]
                                ),
                                style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px", "marginBottom": "20px"}
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Developer Info", style={"fontWeight": "semibold"}),
                                        html.P("Role: Final Year Data Science Student", className="m-0 text-secondary"),
                                        html.P("Project: Liver Patient Prediction Capstone", className="m-0 text-secondary"),
                                        html.P("Institution: University College Project", className="m-0 text-secondary")
                                    ]
                                ),
                                style={"border": f"1px solid {COLORS['borders']}", "borderRadius": "6px"}
                            )
                        ],
                        md=4
                    )
                ]
            )
        ]
    )

# ------------------------------------------------------------------
# DASH CALLBACKS
# ------------------------------------------------------------------

# URL router callback
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page(pathname):
    if not data_loaded:
        return html.Div(
            dbc.Alert(
                f"Data and models could not be initialized. Please check error message: {error_message}",
                color="danger"
            )
        )
    
    if pathname == "/" or pathname == "/overview":
        return render_overview_page()
    elif pathname == "/dataset-analysis":
        return render_dataset_analysis_page()
    elif pathname == "/feature-relationships":
        return render_feature_relationships_page()
    elif pathname == "/model-performance":
        return render_model_performance_page()
    elif pathname == "/prediction-tool":
        return render_prediction_tool_page()
    elif pathname == "/about-project":
        return render_about_page()
    else:
        return html.Div(
            [
                html.H2("404: Not found", className="text-danger"),
                html.P(f"The pathname {pathname} was not recognized.")
            ]
        )

# Feature distribution callback (Dataset analysis page)
@app.callback(
    [
        Output('analysis-dist-plot', 'figure'),
        Output('analysis-box-plot', 'figure')
    ],
    Input('analysis-feature-dropdown', 'value')
)
def update_analysis_plots(selected_col):
    # Histogram with KDE approximation style using Plotly
    fig_hist = px.histogram(
        df,
        x=selected_col,
        marginal="rug",
        color_discrete_sequence=['teal'],
        title=f"{selected_col} Distribution (Histogram & Rug)"
    )
    fig_hist.update_layout(margin=dict(l=30, r=20, t=40, b=30), height=350)
    fig_hist.layout.template = 'plotly_white'
    
    # Boxplot grouped by target matching the notebook
    # df['Dataset'] = 1 (Patient), 2 (Healthy)
    fig_box = px.box(

        df,
        x='Dataset',
        y=selected_col,
        color='Dataset',
        color_discrete_map={1: COLORS['danger'], 2: COLORS['success']},
        title=f"{selected_col} Boxplot Grouped by Target"
    )
    fig_box.update_layout(margin=dict(l=30, r=20, t=40, b=30), height=350, showlegend=False)
    fig_box.layout.template = 'plotly_white'
    
    return fig_hist, fig_box

# Scatter plot callback (Feature relationships page)
@app.callback(
    Output('relationship-scatter-plot', 'figure'),
    [
        Input('x-feature-dropdown', 'value'),
        Input('y-feature-dropdown', 'value'),
        Input('gender-filter-dropdown', 'value'),
        Input('target-filter-dropdown', 'value')
    ]
)
def update_scatter_plot(x_col, y_col, gender_val, target_val):
    filtered_df = df.copy()
    
    # Gender filter
    if gender_val != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == gender_val]
        
    # Target filter
    if target_val != 'All':
        filtered_df = filtered_df[filtered_df['Dataset'] == target_val]
        
    fig = px.scatter(
        filtered_df,
        x=x_col,
        y=y_col,
        color='Dataset',
        color_continuous_scale=[COLORS['danger'], COLORS['success']],
        labels={'Dataset': 'Class'},
        title=f"{x_col} vs {y_col} Relation"
    )
    
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=30), height=450)
    fig.layout.template = 'plotly_white'
    return fig

# Diagnosis predictor callback
@app.callback(
    Output("prediction-result-display", "children"),
    Input("btn-predict", "n_clicks"),
    [
        State("input-age", "value"),
        State("input-gender", "value"),
        State("input-tb", "value"),
        State("input-db", "value"),
        State("input-ap", "value"),
        State("input-sgpt", "value"),
        State("input-sgot", "value"),
        State("input-tp", "value"),
        State("input-alb", "value"),
        State("input-ag", "value")
    ]
)
def run_prediction(n_clicks, age, gender, tb, db, ap, sgpt, sgot, tp, alb, ag):
    if n_clicks is None or n_clicks == 0:
        return dbc.Alert("Enter patient details and click Predict to see the diagnosis results here.", color="info")
    
    # Data Validation
    inputs = [age, gender, tb, db, ap, sgpt, sgot, tp, alb, ag]
    if any(val is None for val in inputs):
        return dbc.Alert("Please fill in all input fields correctly before predicting.", color="warning")
    
    try:
        # Encode gender using trained label encoder if present
        gender_encoded = le.transform([gender])[0]
        
        # Prepare inputs as DataFrame to match fit feature names
        input_data = pd.DataFrame([[age, gender_encoded, tb, db, ap, sgpt, sgot, tp, alb, ag]], 
                                  columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                                           'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                                           'Aspartate_Aminotransferase', 'Total_Protiens', 
                                           'Albumin', 'Albumin_and_Globulin_Ratio'])
        
        # Scaling inputs
        input_scaled = scaler.transform(input_data)
        
        # Run prediction using the best trained model (Random Forest)
        prediction = best_model.predict(input_scaled)[0]
        probabilities = best_model.predict_proba(input_scaled)[0]
        
        # Since target mapping is 1 for Patient, 0 for Healthy
        if prediction == 1:
            confidence = probabilities[1] * 100
            return html.Div(
                [
                    dbc.Alert(
                        [
                            html.H5("Diagnosis: LIVER DISEASE DETECTED", style={"fontWeight": "bold"}),
                            html.P(f"The model classifies this profile as a Liver Patient with a confidence score of {confidence:.2f}%.", className="mb-0")
                        ],
                        color="danger"
                    ),
                    html.P("Recommendation: Clinical verification and consulting a medical professional is advised.", className="text-muted mt-2", style={"fontSize": "0.85rem"})
                ]
            )
        else:
            confidence = probabilities[0] * 100
            return html.Div(
                [
                    dbc.Alert(
                        [
                            html.H5("Diagnosis: NO LIVER DISEASE DETECTED", style={"fontWeight": "bold"}),
                            html.P(f"The model classifies this profile as Healthy with a confidence score of {confidence:.2f}%.", className="mb-0")
                        ],
                        color="success"
                    ),
                    html.P("Recommendation: Maintain a healthy lifestyle and proceed with regular medical checkups.", className="text-muted mt-2", style={"fontSize": "0.85rem"})
                ]
            )
    except Exception as ex:
        return dbc.Alert(f"Prediction failed: {str(ex)}", color="danger")

# ------------------------------------------------------------------
# APP INITIALIZATION RUNNER
# ------------------------------------------------------------------

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == "__main__":
    # Open local browser automatically in 1 second
    Timer(1, open_browser).start()
    
    # Start web server
    app.run(debug=False, port=8050)


