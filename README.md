# Customer Segmentation & RFM Analysis

A data science project demonstrating customer segmentation using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering.

## Project Overview

This project analyzes customer transaction data to:
- Calculate RFM scores for customer value assessment
- Apply K-Means clustering to identify distinct customer personas
- Generate insights for targeted marketing strategies

## Project Structure

```
├── data/
│   ├── raw/                 # Original transaction data
│   └── processed/           # RFM scores and segments
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_rfm_analysis.ipynb
│   └── 04_clustering.ipynb
├── src/                     # Python modules
├── powerbi/                 # Power BI dashboard
├── reports/figures/         # Visualizations
└── tests/                   # Unit tests
```

## Customer Segments

| Segment | Description |
|---------|-------------|
| Champions | Best customers: high recency, frequency, and monetary |
| Loyal Customers | Regular, consistent purchasers |
| Potential Loyalists | Recent customers with good frequency |
| At Risk | Previously valuable, now declining activity |
| Hibernating | Low activity across all metrics |
| New Customers | Recent first-time buyers |

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run notebooks in order (01 → 04)
3. Open Power BI dashboard for visualizations

## Technologies

- Python (pandas, scikit-learn, matplotlib, seaborn, plotly)
- Jupyter Notebooks
- Power BI
