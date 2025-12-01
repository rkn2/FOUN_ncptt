import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Configuration
INPUT_FILE = 'defunct/2023_12_8_targeted_eval.csv'
FIGURE_DIR = 'journalPaper/Images'
RANDOM_STATE = 42

def load_and_prep_data():
    df = pd.read_csv(INPUT_FILE)
    
    # Handle Dual Elevation Measurements (Average them)
    if 'foundation height 2' in df.columns:
        df['Foundation Height'] = df[['foundation height', 'foundation height 2']].mean(axis=1)
    else:
        df['Foundation Height'] = df['foundation height']
        
    if 'ele 2 foundation displ' in df.columns:
        df['Foundation Displacement'] = df[['elev 1 foundation disp', 'ele 2 foundation displ']].mean(axis=1)
    else:
        df['Foundation Displacement'] = df['elev 1 foundation disp']
        
    if 'ele 2 foundation mortar condition' in df.columns:
        df['Foundation Mortar Condition'] = df[['ele 1 foundation mortar condition', 'ele 2 foundation mortar condition']].mean(axis=1)
    else:
        df['Foundation Mortar Condition'] = df['ele 1 foundation mortar condition']

    # Column Mapping
    column_mapping = {
        'Section ': 'WallID',
        'Orientation': 'Orientation',
        'Height': 'Height',
        'Length': 'Length',
        'Thickness': 'Thickness',
        'Material': 'Material',
        '# Courses': 'Courses',
        'Total Scr': 'Total Score',
        'Coat 1 Cracking': 'Coat 1 Cracking',
        'Coat 1 Loss': 'Coat 1 Loss',
        'Coat2 Cracking': 'Coat 2 Cracking',
        'Coat2 Loss': 'Coat 2 Loss',
        'Structural Cracking': 'Structural Cracking',
        'Cracking Wall Junction': 'Cracking at Wall Junction',
        'Lintel Deteriration': 'Lintel Deterioration',
        'Sill 1': 'Sill 1',
        'SILL 2': 'Sill 2',
        'Cap Deterioration': 'Cap Deterioration',
        'Out of Plane': 'Out of Plane',
        'Animal Activity': 'Animal Activity',
        'fireplace': 'Fireplace',
        'Bracing': 'Bracing',
        'Bracing score': 'Bracing Score',
        'PC mean': 'Point Cloud Mean',
        'PC deviation': 'Point Cloud Deviation',
        'Foundation Stone Deterioration': 'Foundation Stone Deterioration'
    }
    
    df_renamed = df.rename(columns=column_mapping)
    
    # Create Treatment variable
    treat_cols = [c for c in df.columns if 'treatment_' in c]
    if treat_cols:
        df_renamed['Treatment'] = df[treat_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    else:
        df_renamed['Treatment'] = 0

    return df_renamed

def get_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=RANDOM_STATE),
        'Elastic Net': ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
        'Decision Tree': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'SVR': SVR(kernel='rbf'),
        'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE)
    }
    return models

def run_benchmark(X, y):
    models = get_models()
    results = {}
    raw_scores = {}
    
    # Repeated K-Fold: 5 folds, 5 repeats = 25 runs
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    
    print("Running Benchmark (5 folds x 5 repeats)...")
    
    for name, model in models.items():
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=rkf, scoring='r2')
        results[name] = {
            'mean_r2': np.mean(scores),
            'std_r2': np.std(scores)
        }
        raw_scores[name] = scores
        print(f"{name}: R2 = {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
        
    return results, raw_scores

def perform_statistical_tests(raw_scores):
    # Identify best model
    means = {k: np.mean(v) for k, v in raw_scores.items()}
    best_model_name = max(means, key=means.get)
    best_scores = raw_scores[best_model_name]
    
    print(f"\nBest Model: {best_model_name} (R2 = {means[best_model_name]:.3f})")
    print("\nStatistical Significance Tests (Wilcoxon Signed-Rank + Holm-Bonferroni):")
    
    p_values = []
    comparisons = []
    
    for name, scores in raw_scores.items():
        if name == best_model_name:
            continue
            
        # Wilcoxon signed-rank test
        stat, p = wilcoxon(best_scores, scores, alternative='two-sided')
        p_values.append(p)
        comparisons.append(name)
        
    # Manual Holm-Bonferroni Correction
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    sorted_names = np.array(comparisons)[sorted_indices]
    
    m = len(p_values)
    reject = []
    p_corrected = []
    
    for i, p in enumerate(sorted_p):
        # Holm-Bonferroni formula: p * (m - i)
        # Must be monotonic, so take max of current and previous
        adj_p = p * (m - i)
        adj_p = min(adj_p, 1.0) # Cap at 1.0
        if i > 0:
            adj_p = max(adj_p, p_corrected[-1])
            
        p_corrected.append(adj_p)
        reject.append(adj_p < 0.05)
        
    # Map back to original order
    final_results = {}
    for i, idx in enumerate(sorted_indices):
        name = comparisons[idx]
        is_equiv = not reject[i]
        final_results[name] = {
            'p_raw': p_values[idx],
            'p_corrected': p_corrected[i],
            'equivalent_to_best': is_equiv
        }
        status = "Indistinguishable" if is_equiv else "Significantly Worse"
        print(f"{name} vs {best_model_name}: p_adj={p_corrected[i]:.4f} -> {status}")
        
    return best_model_name, final_results

def plot_benchmark_results(raw_scores, best_model_name, stats_results):
    df_plot = pd.DataFrame(raw_scores)
    
    # Order by mean score
    means = df_plot.mean().sort_values(ascending=False)
    df_plot = df_plot[means.index]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, palette="viridis")
    plt.title("Regression Model Benchmark (25 Evaluation Rounds)")
    plt.ylabel("R² Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/model_benchmark_boxplot.png', dpi=300)
    plt.close()

def generate_latex_tables(results, raw_scores, best_model_name, stats_results):
    """Generate LaTeX tables for main text and supplementary material"""
    
    # Main Text Table: Top 4 Models
    print("\n=== LaTeX Table for Main Text (Top 4 Models) ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Model Benchmarking Results (Repeated K-Fold Cross-Validation)}")
    print("\\label{tab:model_benchmark}")
    print("\\begin{tabular}{lccl}")
    print("\\hline")
    print("Model & Mean $R^2$ & Std Dev & Status vs Best \\\\")
    print("\\hline")
    
    # Sort by mean R2
    sorted_models = sorted(results.items(), key=lambda x: x[1]['mean_r2'], reverse=True)
    
    for i, (name, res) in enumerate(sorted_models[:4]):
        if name == best_model_name:
            status = "Best"
        elif name in stats_results:
            if stats_results[name]['equivalent_to_best']:
                status = f"$p={stats_results[name]['p_corrected']:.2f}$"
            else:
                status = f"$p<0.001$"
        else:
            status = "N/A"
        
        print(f"{name} & {res['mean_r2']:.3f} & $\\pm${res['std_r2']:.3f} & {status} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Supplementary Table: Full Results
    print("\n=== LaTeX Table for Supplementary Material (All Models) ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Complete Model Benchmarking Results with Statistical Comparisons}")
    print("\\label{tab:model_benchmark_full}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Model & Mean $R^2$ & Std Dev & $p_{raw}$ vs Best & $p_{adj}$ (Holm) \\\\")
    print("\\hline")
    
    for name, res in sorted_models:
        if name == best_model_name:
            p_raw = "---"
            p_adj = "---"
        elif name in stats_results:
            p_raw = f"{stats_results[name]['p_raw']:.4f}"
            p_adj = f"{stats_results[name]['p_corrected']:.4f}"
        else:
            p_raw = "N/A"
            p_adj = "N/A"
        
        print(f"{name} & {res['mean_r2']:.3f} & $\\pm${res['std_r2']:.3f} & {p_raw} & {p_adj} \\\\")
    
    print("\\hline")
    print(f"\\multicolumn{{5}}{{l}}{{\\footnotesize Best model: {best_model_name} ($R^2$ = {results[best_model_name]['mean_r2']:.3f})}} \\\\")
    print("\\multicolumn{5}{l}{\\footnotesize Wilcoxon signed-rank test with Holm-Bonferroni correction ($\\alpha=0.05$)} \\\\")
    print("\\end{tabular}")
    print("\\end{table}")

def main():
    df = load_and_prep_data()
    
    feature_cols = [
        'Height', 'Foundation Height', 'Treatment', 'Bracing', 'Bracing Score',
        'Animal Activity', 'Fireplace', 'Point Cloud Mean', 'Point Cloud Deviation',
        'Sill 1', 'Sill 2', 'Coat 1 Cracking', 'Coat 1 Loss',
        'Lintel Deterioration', 'Coat 2 Cracking', 'Coat 2 Loss',
        'Structural Cracking', 'Cracking at Wall Junction', 'Out of Plane',
        'Cap Deterioration', 'Foundation Stone Deterioration',
        'Foundation Displacement', 'Foundation Mortar Condition'
    ]
    
    existing_cols = [c for c in feature_cols if c in df.columns]
    X = df[existing_cols]
    y = df['Total Score']
    
    results, raw_scores = run_benchmark(X, y)
    best_model, stats = perform_statistical_tests(raw_scores)
    plot_benchmark_results(raw_scores, best_model, stats)
    generate_latex_tables(results, raw_scores, best_model, stats)

if __name__ == "__main__":
    main()
