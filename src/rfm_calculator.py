"""
RFM Calculator Module
Calculates Recency, Frequency, Monetary metrics and assigns customer segments.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def calculate_rfm(
    transactions_df: pd.DataFrame,
    customer_id_col: str = 'customer_id',
    date_col: str = 'transaction_date',
    amount_col: str = 'total_amount',
    reference_date: str = None
) -> pd.DataFrame:
    """
    Calculate RFM metrics for each customer.
    
    Parameters:
    -----------
    transactions_df : DataFrame with transaction data
    customer_id_col : Column name for customer ID
    date_col : Column name for transaction date
    amount_col : Column name for transaction amount
    reference_date : Reference date for recency calculation (default: max date in data)
    
    Returns:
    --------
    DataFrame with RFM metrics per customer
    """
    
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Set reference date
    if reference_date is None:
        ref_date = df[date_col].max()
    else:
        ref_date = pd.to_datetime(reference_date)
    
    # Calculate RFM metrics
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (ref_date - x.max()).days,  # Recency
        'transaction_id': 'count',                       # Frequency
        amount_col: 'sum'                                # Monetary
    }).reset_index()
    
    rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
    
    # Add additional metrics
    rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
    
    return rfm


def assign_rfm_scores(
    rfm_df: pd.DataFrame,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Assign RFM scores using quintile-based scoring (1-5).
    
    Parameters:
    -----------
    rfm_df : DataFrame with RFM metrics
    n_quantiles : Number of quantiles for scoring (default: 5)
    
    Returns:
    --------
    DataFrame with RFM scores added
    """
    
    df = rfm_df.copy()
    
    # For Recency: lower is better, so we reverse the labels
    df['R_score'] = pd.qcut(
        df['recency'].rank(method='first'), 
        q=n_quantiles, 
        labels=range(n_quantiles, 0, -1)
    ).astype(int)
    
    # For Frequency and Monetary: higher is better
    df['F_score'] = pd.qcut(
        df['frequency'].rank(method='first'), 
        q=n_quantiles, 
        labels=range(1, n_quantiles + 1)
    ).astype(int)
    
    df['M_score'] = pd.qcut(
        df['monetary'].rank(method='first'), 
        q=n_quantiles, 
        labels=range(1, n_quantiles + 1)
    ).astype(int)
    
    # Create combined RFM score string
    df['RFM_score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
    
    # Create weighted composite score (can be adjusted)
    df['RFM_composite'] = (df['R_score'] * 0.30 + 
                           df['F_score'] * 0.35 + 
                           df['M_score'] * 0.35)
    
    return df


def assign_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign customer segments based on RFM scores.
    
    Segment logic based on R, F, M scores:
    - Champions: Best in all dimensions
    - Loyal Customers: High F and M, good R
    - Potential Loyalists: Good R and F
    - New Customers: High R, low F
    - At Risk: Declining R with good history
    - Can't Lose Them: Low R but high F and M
    - Hibernating: Low across all dimensions
    - Lost: Very low across all dimensions
    
    Returns:
    --------
    DataFrame with segment column added
    """
    
    df = rfm_df.copy()
    
    def segment_customer(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        
        # Champions: Top in all dimensions
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        
        # Loyal Customers: High frequency and monetary, decent recency
        elif f >= 4 and m >= 4:
            return 'Loyal Customers'
        
        # Potential Loyalists: Recent with moderate frequency
        elif r >= 4 and f >= 3:
            return 'Potential Loyalists'
        
        # New Customers: Very recent, low frequency
        elif r >= 4 and f <= 2:
            return 'New Customers'
        
        # At Risk: Previously good, declining recency
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        
        # Can't Lose Them: Was very valuable, now declining
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        
        # Needs Attention: Average customers showing decline
        elif r == 3 and f == 3:
            return 'Needs Attention'
        
        # About to Sleep: Below average but not gone
        elif r <= 2 and f <= 2 and m >= 2:
            return 'About to Sleep'
        
        # Hibernating: Very low activity
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        
        # Default: Regular customers
        else:
            return 'Regular'
    
    df['segment'] = df.apply(segment_customer, axis=1)
    
    return df


def get_segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by segment."""
    
    summary = rfm_df.groupby('segment').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': ['mean', 'sum'],
        'avg_order_value': 'mean',
        'RFM_composite': 'mean'
    }).round(2)
    
    summary.columns = ['customer_count', 'avg_recency', 'avg_frequency', 
                       'avg_monetary', 'total_revenue', 'avg_order_value', 
                       'avg_rfm_score']
    
    summary['revenue_pct'] = (summary['total_revenue'] / summary['total_revenue'].sum() * 100).round(1)
    summary['customer_pct'] = (summary['customer_count'] / summary['customer_count'].sum() * 100).round(1)
    
    return summary.sort_values('total_revenue', ascending=False)


def run_rfm_analysis(
    transactions_df: pd.DataFrame,
    save_path: str = None
) -> tuple:
    """
    Run complete RFM analysis pipeline.
    
    Returns:
    --------
    Tuple of (rfm_df, segment_summary)
    """
    
    print("Calculating RFM metrics...")
    rfm = calculate_rfm(transactions_df)
    
    print("Assigning RFM scores...")
    rfm = assign_rfm_scores(rfm)
    
    print("Segmenting customers...")
    rfm = assign_segments(rfm)
    
    print("Generating segment summary...")
    summary = get_segment_summary(rfm)
    
    print(f"\nRFM Analysis Complete:")
    print(f"  Total customers: {len(rfm)}")
    print(f"  Segments identified: {rfm['segment'].nunique()}")
    
    if save_path:
        rfm.to_csv(save_path, index=False)
        print(f"  Saved to: {save_path}")
    
    return rfm, summary


if __name__ == "__main__":
    # Test with sample data
    transactions = pd.read_csv("data/raw/ecommerce_transactions.csv")
    rfm_df, summary = run_rfm_analysis(
        transactions, 
        save_path="data/processed/rfm_scores.csv"
    )
    print("\nSegment Summary:")
    print(summary)
