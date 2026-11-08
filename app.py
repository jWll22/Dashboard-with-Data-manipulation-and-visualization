import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for server-side image generation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback_context
import dash_bootstrap_components as dbc
import itertools
import webbrowser


# Create assets folder for saved charts (used by app.py dashboard)
os.makedirs("assets", exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Fix BOM character that Excel adds to the first column name
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# First 10 rows:
print("First 10 rows: \n", df.head(10))

# Last 5 rows":
print("\nLast 5 rows: \n", df.tail(5))

# Shape of the Columns and Rows:
print("\nShape of the dataset: \n", df.shape)


# Display column names, data types, and non-null counts:
# FIX: df.info is a bound method — must be called with () to print actual output
print("\nColumn Names | Data Types | Non-Null Counts: \n")
df.info()

# Generate summary statistics for ALL columns (include=all):
print("\nSummary of all columns: \n", df.describe(include="all"))

# Identify and list all unique values for: Ship Mode, Segment, Region, Category:
print("\nShip Mode:", df["Ship Mode"].unique())
print("\nSegment:", df["Segment"].unique())
print("\nRegion:", df["Region"].unique())
print("\nCategory:", df["Category"].unique())

# Calculate the total number of missing values per column
cal_total_num = df.isnull().sum()

# Calculate the percentage of missing values per column:
cal_pcntage_val = (cal_total_num / len(df) * 100).round(4)

summary_of_missingValues = pd.DataFrame({
    "Total Number of missing values": cal_total_num,
    "Total Percentage of missing values": cal_pcntage_val
})

print("\nMissing values:\n", summary_of_missingValues.to_string())
print(f"\nTotal Missing values across the dataframe:{summary_of_missingValues.sum()}\n")

# Bar graph showing null values:
plt.figure(figsize=(13, 5))
colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in cal_total_num.values]
bars = plt.bar(cal_total_num.index, cal_total_num.values,
               color=colors, edgecolor="white", linewidth=0.8)
plt.title("Missing Values per Column (Green = None)", fontsize=14, fontweight="bold")
plt.xlabel("Column", fontsize=11)
plt.ylabel("Number of Missing Values", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             str(int(bar.get_height())),
             ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("assets/missing_values.png", dpi=150, bbox_inches="tight")

print("Saved: assets/missing_values.png")

'''Documentation: There are no missing values or null values in the dataset,
   in conclusion there is no need for dropping or filling values.
   The bar chart above confirms every column has 0 missing values.
'''

# Check for and count duplicate rows based on ALL columns:
cnt_dup_rows = df.duplicated().sum()
print("\nCount Duplicates:\n", cnt_dup_rows)

# Check for duplicate Order IDs that might represent legitimate multiple items:
dup_orderID = df["Order ID"].value_counts()

multiple_itemOrders = dup_orderID[dup_orderID > 1]
print("\nOrder ID's:\n", multiple_itemOrders)

print("\nOrder ID's with multiple items:\n", len(multiple_itemOrders))

# Remove any true duplicates and document how many were removed:

'''Duplicated before removal: '''
before_removal = len(df)

df = df.drop_duplicates()

'''Duplicates after removal: '''
after_removal = len(df)

'''Calculation of Duplicated rows removed: '''
duplicates_dropped_from_dataset = before_removal - after_removal

print("\nRows before removal:\n", before_removal)
print("\nRows after removal:\n", after_removal)
print("\nDuplication of rows removed\n", duplicates_dropped_from_dataset)

# 2.3 Data Type Conversion
# Convert Order Date and Ship Date to datetime format:

df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", dayfirst=False)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="mixed", dayfirst=False)

print("\nConversion of Order Date and Ship Date:\n", df[["Order Date", "Ship Date"]].dtypes)

# Create a new column: Shipping_Days = Ship Date - Order Date (as integer days):
df["Shipping_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
print(f"Shipping_Days range: {df['Shipping_Days'].min()} - {df['Shipping_Days'].max()} days "
      f"(mean: {df['Shipping_Days'].mean():.2f})\n")

# Create a new column: Profit_Margin = (Profit / Sales) × 100:
df["Profit_Margin"] = (df["Profit"] / df["Sales"]) * 100
print(f"Profit_Margin range: {df['Profit_Margin'].min():.2f}% "
      f"{df['Profit_Margin'].max():.2f}%\n")

# Extract Year and Month from Order Date as separate columns:
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

print(df[["Order Date", "Year", "Month"]].head())

# Outlier Detection and Treatment:

# Calculate the IQR for the Sales, Profit, and Discount columns
columns = ["Sales", "Profit", "Discount"]

# Store outlier counts for box plot annotations
iqr_info = {}
for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    iqr_info[col] = len(outliers)

    print(f"Columns: {col}\n")
    print(f"Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")
    print(f"Lower Bound={lower_bound:.4f}, Upper Bound={upper_bound:.4f}")
    print(f"Outliers: {len(outliers)}\n")

# Create box plots showing the distribution of Sales and Profit (before any treatment):

# Sales box plot:
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.boxplot(
    y=df["Sales"],
    color="skyblue",
    linewidth=2,
    flierprops={
        "marker": "*",
        "markerfacecolor": "red",
        "markersize": 5,
        "linestyle": "none"
    }
)
plt.title("Distribution of Sales", fontsize=14)
plt.ylabel("Sales Amount ($)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.text(0.97, 0.97, f"Outliers: {iqr_info['Sales']}",
         transform=plt.gca().transAxes, ha="right", va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"))

# Profit box plot:
plt.subplot(1, 2, 2)
sns.boxplot(
    y=df["Profit"],
    color="lightcoral",
    linewidth=2,
    flierprops={
        "marker": "D",
        "markerfacecolor": "darkred",
        "markersize": 5,
        "linestyle": "none"
    }
)
plt.title("Distribution of Profit", fontsize=14, fontweight="bold")
plt.ylabel("Profit ($)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.text(0.97, 0.97, f"Outliers: {iqr_info['Profit']}",
         transform=plt.gca().transAxes, ha="right", va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"))

plt.suptitle("Box Plots of Sales and Profit (Before Outlier Treatment)",
             fontsize=16, fontweight="bold")

plt.tight_layout()
plt.savefig("assets/boxplots.png", dpi=150, bbox_inches="tight")

print("Saved: assets/boxplots.png")


# Exploratory Data Analysis

# Descriptive Statistics with NumPy:
sales_from_df = df["Sales"].to_numpy()
profit_from_df = df["Profit"].to_numpy()

for name, data in [("Sales", sales_from_df), ("Profit", profit_from_df)]:

    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)

    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)

    minimum = np.min(data)
    maximum = np.max(data)
    data_range = maximum - minimum

    print(f"\n{name} Statistics:\n")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"25th Percentile: {p25:.2f}")
    print(f"50th Percentile: {p50:.2f}")
    print(f"75th Percentile: {p75:.2f}")
    print(f"Minimum: {minimum:.2f}")
    print(f"Maximum: {maximum:.2f}")
    print(f"Range: {data_range:.2f}")

    if mean > median:
        skew = "Right-side skewed (positively skewed)"
    elif mean < median:
        skew = "Left-side skewed (negatively skewed)"
    else:
        skew = "Symmetric"

    print(f"Distribution: {skew}")

# Total Sales, Total Profit, and Average Discount by Category:
category_summary = df.groupby("Category").agg({
    "Sales": "sum",
    "Profit": "sum",
    "Discount": "mean"
})

print("\nTotal Sales, Profit and Average Discount:\n", category_summary)

# Total Sales and Profit by Region and Segment (multi-level grouping):
region_segment_summary = df.groupby(
    ["Region", "Segment"]
).agg({
    "Sales": "sum",
    "Profit": "sum"
})

print("\nTotal Sales and Profit by Region and Segment:\n", region_segment_summary)

# Average Profit Margin by Sub-Category, sorted from lowest to highest:
df["Profit_Margin"] = (df["Profit"] / df["Sales"]) * 100

profit_margin_summary = (
    df.groupby("Sub-Category")["Profit_Margin"]
    .mean()
    .sort_values()
)

print("\nAverage Profit Margin by Sub-Category (lowest to highest):")
print(profit_margin_summary)

# Monthly Sales trend (aggregate sales by Year-Month):
monthly_sales = df.groupby(
    ["Year", "Month"]
)["Sales"].sum()

print(monthly_sales)

# Year-Month column:
df["Year_Month"] = df["Order Date"].dt.to_period("M")

monthly_sales = df.groupby("Year_Month")["Sales"].sum()

print(monthly_sales)

correlation_matrix = df[["Sales", "Quantity", "Discount", "Profit"]].corr()

print("\nCorrelation Matrix:\n")
print(correlation_matrix)

# Identify strongest positive and negative correlation pairs
corr_cols = ["Sales", "Quantity", "Discount", "Profit"]
pairs = [
    (c1, c2, correlation_matrix.loc[c1, c2])
    for c1, c2 in itertools.combinations(corr_cols, 2)
]
pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
print(f"\nStrongest POSITIVE: {pairs_sorted[0][0]} & {pairs_sorted[0][1]}"
      f"  r = {pairs_sorted[0][2]:.4f}")
print(f"Strongest NEGATIVE: {pairs_sorted[-1][0]} & {pairs_sorted[-1][1]}"
      f"  r = {pairs_sorted[-1][2]:.4f}")

# Interpretation of Discount vs Profit
print("""
Discount vs Profit Interpretation (r = -0.2195):
  A meaningful negative correlation — as Discount increases, Profit decreases.
  Notably Discount vs Sales (r = -0.028) is near zero, meaning discounting
  does NOT compensate with higher sales volume. The company is reducing
  prices without gaining offsetting revenue, directly eroding margins.
  A discount threshold of ~40% is where most transactions become loss-making.
""")

# Correlation heatmap (saved to assets for dashboard):
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Matrix of Sales, Quantity, Discount and Profit",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/correlation_heatmap.png", dpi=150, bbox_inches="tight")

print("Saved: assets/correlation_heatmap.png")


# Filter and display all transactions where Profit < 0 (losses):
loss_transactions = df[df["Profit"] < 0]

# Show first 10 rows only — printing all 1,871 rows floods the console
print("\nSample of loss-making transactions (first 10 rows):")
print(loss_transactions.head(10).to_string())

# Total loss amount
total_loss_amount = loss_transactions["Profit"].sum()

# Number of loss-making transactions
num_loss_transactions = len(loss_transactions)

print("\nTotal Loss Amount:", total_loss_amount)
print("Number of Loss-Making Transactions:", num_loss_transactions)

# Identify which Sub-Category has the highest total losses:
subcategoryLosses = loss_transactions.groupby(
    "Sub-Category"
)["Profit"].sum()

print("\nSub-Category Losses (sorted):")
print(subcategoryLosses.sort_values())

# Sub-Category with highest total losses
worst_subcategory = subcategoryLosses.idxmin()
worst_loss = subcategoryLosses.min()

print("Sub-Category with Highest Losses:", worst_subcategory)
print("Total Loss:", worst_loss)

# ASSETS CHARTS
'''Three static charts required by the dashboard (app.py Task 4).
Saved as PNG files to the assets/ folder
'''

print("\nGenerating and saving charts to assets/ folder:\n")


# Assets Folder Chart 1: Profit distribution histogram with KDE
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["Profit"], bins=80, kde=True, color="#3498db",
             edgecolor="white", linewidth=0.4, ax=ax)
ax.axvline(x=0, color="#e74c3c", linewidth=2, linestyle="--",
           label="Break-Even (Profit = $0)")
ax.set_title("Profit Distribution with KDE Overlay", fontsize=14, fontweight="bold")
ax.set_xlabel("Profit (USD)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("assets/profit_histogram.png", dpi=150, bbox_inches="tight")

print("Saved: assets/profit_histogram.png")

# Assets Folder Chart 2: Avg Profit Margin by Sub-Category (horizontal bar):
subcat_avg_margin = df.groupby("Sub-Category")["Profit_Margin"].mean().sort_values()
bar_colors = ["#e74c3c" if v < 0 else "#27ae60" for v in subcat_avg_margin.values]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(subcat_avg_margin.index, subcat_avg_margin.values,
               color=bar_colors, edgecolor="white", linewidth=0.5)
ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
ax.set_title("Average Profit Margin by Sub-Category\n(Red = Negative / Loss-Making)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Average Profit Margin (%)", fontsize=11)
ax.set_ylabel("Sub-Category", fontsize=11)
for bar, val in zip(bars, subcat_avg_margin.values):
    ax.text(val + (0.4 if val >= 0 else -0.4),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center",
            ha="left" if val >= 0 else "right", fontsize=8)
plt.tight_layout()
plt.savefig("assets/subcategory_margin_bar.png", dpi=150, bbox_inches="tight")

print("Saved: assets/subcategory_margin_bar.png")

# Assets Chart 3: Annotated Seaborn correlation heatmap
# Saving with higher resolution:
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".4f",
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.8,
    annot_kws={"size": 12, "weight": "bold"},
    ax=ax
)
ax.set_title("Correlation Heatmap: Sales, Quantity, Discount, Profit",
             fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("assets/correlation_heatmap.png", dpi=150, bbox_inches="tight")

print("Saved: assets/correlation_heatmap.png (updated with 4dp annotations)")

print("\nAll assets saved. Your assets/ folder now contains:")
for f in sorted(os.listdir("assets")):
    print(f"  assets/{f}")

# Task 4:

sns.set_theme(style="whitegrid", palette="muted")

# Loading CSV file:
# encoding='latin1' handles special characters present in the Superstore file
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Fix BOM character Excel sometimes adds to the first column name
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# Date conversions:
# format='mixed' required because the file mixes M/DD/YYYY and MM/DD/YYYY patterns
df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", dayfirst=False)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="mixed", dayfirst=False)

# Create calculated columns if they do not already exist:
df["Shipping_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

df["Profit_Margin"] = (
    df["Profit"] / df["Sales"]) * 100

df["Year_Month"] = (
    df["Order Date"]
    .dt.to_period("M")
    .astype(str))

df["Year"]  = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# ── Dropdown options (reused in layout and callbacks) ─────────────────────────
ALL_REGIONS    = sorted(df["Region"].unique().tolist())
ALL_CATEGORIES = sorted(df["Category"].unique().tolist())
ALL_SEGMENTS   = sorted(df["Segment"].unique().tolist())

# =============================================================================
# STATIC CHARTS — Matplotlib / Seaborn
# Generated once at startup and embedded as base64 PNG data URIs.
# This is the correct approach for Dash; the assets/ folder is for CSS files
# and any manually placed images, not for programmatically generated charts.
# =============================================================================

def fig_to_b64(fig) -> str:
    """Convert a Matplotlib figure to a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


# Static Chart 1 — Profit distribution histogram with KDE overlay
# Requirement: vertical line at Profit=0, descriptive title, axis labels
def build_profit_histogram() -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["Profit"], bins=80, kde=True, color="#3498db",
                 edgecolor="white", linewidth=0.4, ax=ax)
    ax.axvline(x=0, color="#e74c3c", linewidth=2, linestyle="--",
               label="Break-Even (Profit = $0)")
    ax.set_title("Profit Distribution with KDE Overlay", fontsize=13, fontweight="bold")
    ax.set_xlabel("Profit (USD)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


# Static Chart 2 — Horizontal bar chart: Avg Profit Margin by Sub-Category
# Requirement: negative values highlighted in red, sorted lowest to highest
def build_subcat_bar() -> str:
    subcat_pm = df.groupby("Sub-Category")["Profit_Margin"].mean().sort_values()
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in subcat_pm.values]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(subcat_pm.index, subcat_pm.values,
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_title("Average Profit Margin by Sub-Category\n(Red = Loss-Making)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Average Profit Margin (%)", fontsize=10)
    ax.set_ylabel("Sub-Category", fontsize=10)
    for bar, val in zip(bars, subcat_pm.values):
        ax.text(val + (0.4 if val >= 0 else -0.4),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


# Static Chart 3 — Annotated Seaborn correlation heatmap
# Requirement: numeric variables Sales, Quantity, Discount, Profit
def build_corr_heatmap() -> str:
    corr = df[["Sales", "Quantity", "Discount", "Profit"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".4f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, linewidths=0.8,
                annot_kws={"size": 11, "weight": "bold"}, ax=ax)
    ax.set_title("Correlation Heatmap: Sales, Quantity, Discount, Profit",
                 fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig_to_b64(fig)


# Generate all three static images at startup
IMG_HISTOGRAM  = build_profit_histogram()
IMG_SUBCAT_BAR = build_subcat_bar()
IMG_CORR_HMAP  = build_corr_heatmap()

# Interactive PLotly Charts:

# KPI values (default — unfiltered):
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
avg_profit_margin = df["Profit_Margin"].mean()
num_orders = df["Order ID"].nunique()

# Sales vs Profit Scatter Plot:
scatter_fig = px.scatter(
    df,
    x="Sales",
    y="Profit",
    color="Segment",
    size="Quantity",
    size_max=18,
    hover_data=[
        "Product Name",
        "Category",
        "Sub-Category",
        "Discount",
        "Region"
    ],
    title="Sales vs Profit  |  colour: Segment  |  size: Quantity",
    template="plotly_white"
)
# Break-even line at Profit = 0
scatter_fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c",
                      annotation_text="Break-Even",
                      annotation_position="bottom right")

# Monthly sales trend with range slider:
monthly = (
    df.groupby("Year_Month")
      ["Sales"]
      .sum()
      .reset_index()
)

trend_fig = px.line(
    monthly,
    x="Year_Month",
    y="Sales",
    title="Monthly Sales Trend",
    template="plotly_white"
)
trend_fig.update_layout(
    xaxis=dict(rangeslider=dict(visible=True), type="category"),
    yaxis_tickprefix="$",
    hovermode="x unified"
)

# Regional Sales and Profit bar chart:
region_summary = (
    df.groupby("Region")
      [["Sales","Profit"]]
      .sum()
      .reset_index()
)

region_fig = px.bar(
    region_summary,
    x="Region",
    y=["Sales", "Profit"],
    barmode="group",
    title="Regional Sales and Profit Performance",
    color_discrete_map={"Sales": "#2980b9", "Profit": "#27ae60"},
    template="plotly_white"
)
region_fig.update_layout(yaxis_tickprefix="$")


# *** DASH APPLICATION: ***


# Creating Dash application:
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

app.title = "SuperStore Dashboard"

# LAYOUT
app.layout = dbc.Container([

    # Header
    dbc.Row(dbc.Col(html.Div([
        html.H1("SuperStore Performance Dashboard",
                className="text-center mt-3 mb-1",
                style={"color": "#2C3E50", "fontWeight": "700"}),
        html.H5("",
                className="text-center text-muted mb-1"),
        html.P("Interactive analysis of SuperStore Inc. sales, profit, and regional "
               "performance (2014–2017). Use the filters below to explore the data.",
               className="text-center text-muted",
               style={"fontSize": "0.9rem"}),
        html.Hr()
    ]))),

    # Filters row:
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Region", className="fw-bold small text-muted"),
            dcc.Dropdown(
                id="region_dropdown",
                options=[{"label": r, "value": r} for r in ALL_REGIONS],
                value=ALL_REGIONS,
                multi=True,
                placeholder="All regions"
            )
        ], md=4),

        dbc.Col([
            html.Label("Filter by Category", className="fw-bold small text-muted"),
            dcc.Dropdown(
                id="category_dropdown",
                options=[{"label": c, "value": c} for c in ALL_CATEGORIES],
                value=ALL_CATEGORIES,
                multi=True,
                placeholder="All categories"
            )
        ], md=4),

        dbc.Col([
            html.Label("Filter by Date Range", className="fw-bold small text-muted"),
            dcc.DatePickerRange(
                id="date_picker",
                min_date_allowed=df["Order Date"].min(),
                max_date_allowed=df["Order Date"].max(),
                start_date=df["Order Date"].min(),
                end_date=df["Order Date"].max(),
                display_format="DD/MM/YYYY"
            )
        ], md=4),
    ], className="mb-3"),

    # KPI Cards:
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Total Sales", className="text-muted small mb-1"),
            html.H3(id="kpi_sales", className="fw-bold", style={"color": "#2980b9"})
        ]), style={"borderLeft": "4px solid #2980b9", "borderRadius": "8px"}), md=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Total Profit", className="text-muted small mb-1"),
            html.H3(id="kpi_profit", className="fw-bold", style={"color": "#27ae60"})
        ]), style={"borderLeft": "4px solid #27ae60", "borderRadius": "8px"}), md=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Avg Profit Margin", className="text-muted small mb-1"),
            html.H3(id="kpi_avg_margin", className="fw-bold", style={"color": "#e67e22"})
        ]), style={"borderLeft": "4px solid #e67e22", "borderRadius": "8px"}), md=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Number of Orders", className="text-muted small mb-1"),
            html.H3(id="kpi_num_orders", className="fw-bold", style={"color": "#8e44ad"})
        ]), style={"borderLeft": "4px solid #8e44ad", "borderRadius": "8px"}), md=3),
    ], className="mb-4"),

    # Tabs:
    dbc.Tabs([

        # --- TAB 1: Overview ---
        dbc.Tab(label="Overview", tab_id="tab-overview", children=[
            dbc.Row([
                # Interactive scatter plot (unchanged)
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.Strong("Sales vs Profit Scatter Plot")),
                    dbc.CardBody(dcc.Graph(id="scatter_graph", figure=scatter_fig,
                                           style={"height": "460px"}))
                ], className="shadow-sm mt-3"), md=7),

                # Static histogram (changed to use base64 IMG_HISTOGRAM)
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.Strong("Profit Distribution (Matplotlib)")),
                    dbc.CardBody([
                        html.Img(src=IMG_HISTOGRAM,
                                 style={"width": "100%", "borderRadius": "5px"}),
                        html.P("The profit distribution is right-skewed with a heavy left "
                               "tail. The red dashed line marks the break-even point. "
                               "The concentration of values left of zero reflects the "
                               "impact of deep discounting on Furniture and Binder "
                               "sub-categories.",
                               className="text-muted mt-2",
                               style={"fontSize": "0.8rem"})
                    ])
                ], className="shadow-sm mt-3"), md=5),
            ])
        ]),

        # --- TAB 2: Regional Analysis ---
        dbc.Tab(label="Regional Analysis", tab_id="tab-regional", children=[
            dbc.Row([
                # Interactive regional bar chart (unchanged)
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.Strong("Sales & Profit by Region")),
                    dbc.CardBody(dcc.Graph(id="region_graph", figure=region_fig,
                                           style={"height": "440px"}))
                ], className="shadow-sm mt-3"), md=7),

                # Static correlation heatmap (changed to use base64 IMG_CORR_HMAP)
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.Strong("Correlation Heatmap (Seaborn)")),
                    dbc.CardBody([
                        html.Img(src=IMG_CORR_HMAP,
                                 style={"width": "100%", "borderRadius": "5px"}),
                        html.P("Discount vs Profit (r = -0.22) is the key negative "
                               "relationship — higher discounts reduce profit without "
                               "driving compensating sales volume (r = -0.028). "
                               "Sales vs Profit (r = 0.48) is the strongest positive link.",
                               className="text-muted mt-2",
                               style={"fontSize": "0.8rem"})
                    ])
                ], className="shadow-sm mt-3"), md=5),
            ])
        ]),

        # TAB 3: Product Analysis
        dbc.Tab(label="Product Analysis", tab_id="tab-product", children=[
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.Strong(
                        "Avg Profit Margin by Sub-Category (Matplotlib)")),
                    dbc.CardBody([
                        html.Img(src=IMG_SUBCAT_BAR,
                                 style={"width": "100%", "borderRadius": "5px"}),
                        html.P("Four sub-categories have negative margins, highlighted in red. "
                               "These represent consistent loss-making areas that need "
                               "pricing or sourcing review.",
                               className="text-muted mt-2",
                               style={"fontSize": "0.8rem"})
                    ])
                ], className="shadow-sm mt-3"), md=12),
            ])
        ]),

        # TAB 4: Trends
        dbc.Tab(label="Trends", tab_id="tab-trends", children=[
            dbc.Row(
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.Strong("Monthly Sales Trend  "),
                        dbc.ButtonGroup([
                            dbc.Button("Last 6M",  id="btn_6m",
                                       color="outline-primary", size="sm"),
                            dbc.Button("Last 12M", id="btn_12m",
                                       color="outline-primary", size="sm"),
                            dbc.Button("All Time", id="btn_all",
                                       color="primary",         size="sm"),
                        ], size="sm")
                    ], style={"display": "flex", "justifyContent": "space-between",
                               "alignItems": "center"}),
                    dbc.CardBody(dcc.Graph(id="trend_graph", figure=trend_fig,
                                           style={"height": "460px"}))
                ], className="shadow-sm mt-3"))
            )
        ]),

    ], id="main_tabs", active_tab="tab-overview"),

], fluid=True)

# Callbacks:

@app.callback(
    Output("kpi_sales",      "children"),
    Output("kpi_profit",     "children"),
    Output("kpi_avg_margin", "children"),
    Output("kpi_num_orders", "children"),
    Output("scatter_graph",  "figure"),
    Output("region_graph",   "figure"),
    Input("region_dropdown", "value"),
    Input("category_dropdown", "value"),
    Input("date_picker",     "start_date"),
    Input("date_picker",     "end_date"),
)
def update_charts(selected_regions, selected_categories, start_date, end_date):
    """Update KPI cards and interactive charts when filters change.

    Filters the main dataframe by region, category, and date range,
    then recalculates all KPI values and rebuilds the scatter and
    regional bar charts from the filtered data.
    Handles edge case of empty selections by falling back to full dataset.
    """
    dff = df.copy()

    # Apply region filter (guard against empty list)
    if selected_regions:
        dff = dff[dff["Region"].isin(selected_regions)]

    # Apply category filter (guard against empty list)
    if selected_categories:
        dff = dff[dff["Category"].isin(selected_categories)]

    # Apply date range filter
    if start_date and end_date:
        dff = dff[
            (dff["Order Date"] >= start_date) &
            (dff["Order Date"] <= end_date)
        ]

    # Guard: if filters result in empty dataframe show message
    if dff.empty:
        empty_fig = go.Figure().add_annotation(
            text="No data matches the selected filters.",
            xref="paper", yref="paper", showarrow=False, font=dict(size=14)
        )
        return "$0", "$0", "0.0%", "0", empty_fig, empty_fig

    # Recalculate KPIs from filtered data
    kpi_sales      = f"${dff['Sales'].sum():,.0f}"
    kpi_profit     = f"${dff['Profit'].sum():,.0f}"
    kpi_avg_margin = f"{dff['Profit_Margin'].mean():.2f}%"
    kpi_num_orders = f"{dff['Order ID'].nunique():,}"

    # Rebuild scatter plot from filtered data
    filtered_scatter = px.scatter(
        dff,
        x="Sales",
        y="Profit",
        color="Segment",
        size="Quantity",
        size_max=18,
        hover_data=["Product Name", "Category", "Sub-Category", "Discount", "Region"],
        title="Sales vs Profit  |  colour: Segment  |  size: Quantity",
        template="plotly_white"
    )
    filtered_scatter.add_hline(y=0, line_dash="dash", line_color="#e74c3c",
                               annotation_text="Break-Even",
                               annotation_position="bottom right")

    # Rebuild regional bar chart from filtered data
    region_data = (
        dff.groupby("Region")[["Sales", "Profit"]]
        .sum()
        .reset_index()
    )
    filtered_region = px.bar(
        region_data,
        x="Region",
        y=["Sales", "Profit"],
        barmode="group",
        title="Regional Sales and Profit Performance",
        color_discrete_map={"Sales": "#2980b9", "Profit": "#27ae60"},
        template="plotly_white"
    )
    filtered_region.update_layout(yaxis_tickprefix="$")

    return (kpi_sales, kpi_profit, kpi_avg_margin, kpi_num_orders,
            filtered_scatter, filtered_region)


@app.callback(
    Output("trend_graph", "figure"),
    Input("region_dropdown",   "value"),
    Input("category_dropdown", "value"),
    Input("date_picker",       "start_date"),
    Input("date_picker",       "end_date"),
    Input("btn_6m",  "n_clicks"),
    Input("btn_12m", "n_clicks"),
    Input("btn_all", "n_clicks"),
)
def update_trend(selected_regions, selected_categories, start_date, end_date,
                 n_6m, n_12m, n_all):
    """Update the monthly sales trend chart with filters and preset range buttons.

    Applies sidebar filters then rebuilds the time-series chart.
    The most recently clicked preset button (Last 6M, Last 12M, All Time)
    restricts the x-axis range accordingly.
    """
    dff = df.copy()

    if selected_regions:
        dff = dff[dff["Region"].isin(selected_regions)]
    if selected_categories:
        dff = dff[dff["Category"].isin(selected_categories)]
    if start_date and end_date:
        dff = dff[
            (dff["Order Date"] >= start_date) &
            (dff["Order Date"] <= end_date)
        ]

    if dff.empty:
        return go.Figure().add_annotation(
            text="No data for selected filters.",
            xref="paper", yref="paper", showarrow=False
        )

    monthly_filtered = (
        dff.groupby("Year_Month")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Year_Month")
    )

    fig = px.line(
        monthly_filtered,
        x="Year_Month",
        y="Sales",
        title="Monthly Sales Trend",
        template="plotly_white"
    )

    # Determine which preset button was last clicked
    ctx = callback_context
    x_range = None
    if ctx.triggered:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        all_months = sorted(monthly_filtered["Year_Month"].tolist())
        if trigger_id == "btn_6m" and len(all_months) >= 6:
            x_range = [all_months[-6], all_months[-1]]
        elif trigger_id == "btn_12m" and len(all_months) >= 12:
            x_range = [all_months[-12], all_months[-1]]
        # btn_all or anything else → show everything (x_range stays None)

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="category",
            range=x_range        # None = show all; list = restrict to preset
        ),
        yaxis_tickprefix="$",
        hovermode="x unified"
    )

    return fig


'''ENTRY POINT:'''
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8050/") # opens the dashboard automatically
    app.run(debug=True)