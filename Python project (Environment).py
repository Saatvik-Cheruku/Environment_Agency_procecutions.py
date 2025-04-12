import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams.update({'figure.figsize': (12, 8), 'axes.labelsize': 14, 
                    'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
sns.set_palette(sns.color_palette("viridis", 10))

def load_and_clean_data(file_path):
    print(f"Loading data from: {file_path}")
    df = pd.read_excel(file_path, sheet_name="Environment Agency Prosecutions")
    
    # Clean column names and print info
    print("\nActual columns in the dataset:", *[f"- {col}" for col in df.columns], sep="\n")
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[()]', '', regex=True).str.replace('.', '', regex=False)
    print("\nCleaned columns:", *[f"- {col}" for col in df.columns], sep="\n")
    
    # Date processing and duplicate removal
    df['Date_of_Action'] = pd.to_datetime(df['Date_of_Action'], errors='coerce')
    df['Year'], df['Month'] = df['Date_of_Action'].dt.year, df['Date_of_Action'].dt.month
    
    initial_rows = len(df)
    df = df.drop_duplicates()
    if initial_rows - len(df) > 0:
        print(f"Removed {initial_rows - len(df)} duplicate rows.")
        
    return df

def currency_formatter(x, pos):
    return f"£{x:,.0f}"

def analyze_data(df):
    print(f"\n=== Dataset Summary ===\nTotal prosecutions: {len(df)}")
    
    if 'Date_of_Action' in df.columns and not df['Date_of_Action'].isna().all():
        print(f"Date range: {df['Date_of_Action'].min().strftime('%Y-%m-%d')} to {df['Date_of_Action'].max().strftime('%Y-%m-%d')}")
    
    if 'Industry_Sector' in df.columns:
        print(f"Total unique industry sectors: {df['Industry_Sector'].nunique()}")
    
    # Analyze numeric columns
    numeric_cols = [col for col in ['Fine', 'CICS_Score', 'Costs', 'Cost', 'Legal_Costs'] if col in df.columns]
    if numeric_cols:
        print(f"\n=== Financial/Numeric Summary for {', '.join(numeric_cols)} ===")
        financial_summary = df[numeric_cols].describe().T
        financial_summary['missing'] = df[numeric_cols].isna().sum()
        print(financial_summary)
    
    # Top verdicts
    if 'Verdict' in df.columns:
        print("\n=== Top 5 Verdicts ===")
        print(df['Verdict'].value_counts().head(5))
    else:
        print("\nNo 'Verdict' column found in the dataset.")

def create_visualization(plot_func, df, title):
    try:
        plot_func(df)
    except Exception as e:
        print(f"Error creating {title}: {e}")

def visualize_industry_sectors(df):
    if 'Industry_Sector' not in df.columns:
        print("Cannot create industry sectors visualization: 'Industry_Sector' column not found.")
        return
        
    top_industries = df['Industry_Sector'].value_counts().head(10).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(16, 10))
    
    bars = ax.barh(top_industries.index, top_industries.values, color=sns.color_palette("viridis", 10))
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.02, bar.get_y() + bar.get_height()/2, f"{width:,.0f}", va='center', fontweight='bold')
    
    plt.subplots_adjust(right=0.85, left=0.25)
    ax.set_title('Top 10 Industry Sectors by Prosecution Count', fontweight='bold', pad=20)
    ax.set_xlabel('Number of Prosecutions', labelpad=15)
    plt.show()

def visualize_verdict_distribution(df):
    if 'Verdict' not in df.columns:
        print("Missing 'Verdict' column.")
        return

    verdict_counts = df['Verdict'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    explode = [0.2 if count / verdict_counts.sum() < 0.05 else 0.01 for count in verdict_counts]
    wedges, texts, autotexts = ax.pie(
        verdict_counts, labels=verdict_counts.index, autopct='%1.1f%%', startangle=90,
        colors=sns.color_palette("viridis", len(verdict_counts)), explode=explode, shadow=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}, textprops={'fontsize': 14},
        pctdistance=0.75, labeldistance=1.15
    )
    
    fig.gca().add_artist(plt.Circle((0, 0), 0.50, fc='white'))
    for i, (text, autotext) in enumerate(zip(texts, autotexts)):
        autotext.set_fontweight('bold')
        autotext.set_color('white')
        if verdict_counts.iloc[i] / verdict_counts.sum() * 100 < 5:
            text.set_bbox(dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.legend(
        wedges,
        [f"{verdict} ({count:,}, {count/verdict_counts.sum()*100:.1f}%)" for verdict, count in verdict_counts.items()],
        title="Verdicts", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    ax.set_title("Verdict Distribution", fontweight='bold', fontsize=16, pad=20)
    ax.axis('equal')
    plt.subplots_adjust(right=0.7, left=0.1)
    plt.show()

def visualize_time_trends(df):
    if 'Year' not in df.columns or df['Year'].isna().all():
        print("Cannot create time trends visualization: 'Year' column not found or all values are NaN.")
        return
        
    yearly_counts = df['Year'].value_counts().sort_index()
    
    def plot_trend(data, title, y_label, color, formatter=None):
        fig, ax = plt.subplots(figsize=(16, 8))
        data.plot(marker='o', markersize=10, linewidth=2, ax=ax, color=color)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_ylabel(y_label, labelpad=15)
        ax.set_xlabel('Year', labelpad=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        if formatter:
            ax.yaxis.set_major_formatter(formatter)
        
        for i, (year, value) in enumerate(data.items()):
            label = f"£{value:,.0f}" if formatter else f"{value:,}"
            y_offset = 10 if i % 2 == 0 else -25
            ax.annotate(label, xy=(year, value), xytext=(0, y_offset),
                        textcoords="offset points", ha='center', fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.show()
    
    # Plot yearly counts
    plot_trend(yearly_counts, 'Prosecutions Over Time', 'Number of Prosecutions', '#3498db')
    
    # Plot average fines if available
    if 'Fine' in df.columns and not df['Fine'].isna().all():
        avg_fine = df.groupby('Year')['Fine'].mean()
        plot_trend(avg_fine, 'Average Fine by Year', 'Average Fine (£)', '#e74c3c', 
                  FuncFormatter(currency_formatter))

def visualize_confusion_matrix(df):
    if not {'Industry_Sector','Verdict'}.issubset(df.columns): return print("Missing columns.")
    df = df[df['Industry_Sector'].isin(df['Industry_Sector'].value_counts().head(8).index) & 
            df['Verdict'].isin(df['Verdict'].value_counts().head(6).index)]
    cmap = sns.color_palette("viridis", as_cmap=True)
    for norm, fmt, title, label in [(True,".1%","Proportions","Proportion"),(False,"d","Counts","Cases")]:
        conf = pd.crosstab(df['Industry_Sector'], df['Verdict'], normalize='index' if norm else False)
        fig, ax = plt.subplots(figsize=(14,12))
        sns.heatmap(conf, annot=True, fmt=fmt, cmap=cmap, linewidths=1.5, linecolor='white',
                    cbar_kws={'label':label,'shrink':0.8}, annot_kws={"size":14,"weight":"bold"}, ax=ax)
        for text in ax.texts:
            v = float(text.get_text().replace('%','')) / (100 if fmt==".1%" else 1)
            text.set_color('white' if v > (0.4 if fmt==".1%" else conf.values.max()/2) else 'black')
        if not norm:
            for i, t in enumerate(conf.sum(axis=1)):
                ax.text(len(conf.columns)+.5, i+.5, f"Total: {t:,}", ha='left', va='center',
                        fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=.8, boxstyle='round,pad=0.3'))
        ax.set(title=title, xlabel="Verdict", ylabel="Industry Sector")
        plt.xticks(rotation=45, ha="right", fontsize=12); plt.yticks(fontsize=12)
        plt.tight_layout(pad=3); fig.subplots_adjust(left=0.25); plt.show()

def main():
    file_path = r"C:\Users\saatv\Downloads\Environment_Agency_Prosecutions.xlsx"
    
    try:
        df = load_and_clean_data(file_path)
        analyze_data(df)
        
        print("\nGenerating visualizations...")
        visualizations = [
            (visualize_industry_sectors, "Industry Sectors"),
            (visualize_verdict_distribution, "Verdict Distribution"),
            (visualize_time_trends, "Time Trends"),
            (visualize_confusion_matrix, "Confusion Matrix")
        ]
        
        for viz_func, title in visualizations:
            create_visualization(viz_func, df, title)
        
        print("\nEDA completed successfully!")
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()