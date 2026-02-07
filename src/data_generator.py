"""
Data Generator Module
Generates synthetic e-commerce transaction data for RFM analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


def generate_customers(n_customers: int = 5000) -> pd.DataFrame:
    """Generate customer base with varied purchasing behavior profiles."""
    
    # Define customer behavior types and their proportions
    behavior_profiles = {
        'vip': 0.05,           # 5% - High frequency, high value
        'loyal': 0.15,         # 15% - Regular purchasers
        'regular': 0.30,       # 30% - Average customers
        'occasional': 0.25,    # 25% - Infrequent buyers
        'dormant': 0.15,       # 15% - Very low activity
        'new': 0.10            # 10% - Recent first-time buyers
    }
    
    customers = []
    for i in range(n_customers):
        profile = np.random.choice(
            list(behavior_profiles.keys()),
            p=list(behavior_profiles.values())
        )
        
        customers.append({
            'customer_id': f'CUST_{i+1:05d}',
            'customer_name': fake.name(),
            'email': fake.email(),
            'country': np.random.choice(
                ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia'],
                p=[0.40, 0.15, 0.15, 0.10, 0.10, 0.10]
            ),
            'join_date': fake.date_between(start_date='-3y', end_date='-30d'),
            'behavior_profile': profile
        })
    
    return pd.DataFrame(customers)


def generate_transactions(
    customers_df: pd.DataFrame,
    start_date: str = '2024-02-01',
    end_date: str = '2026-02-06',
    avg_transactions_per_customer: int = 10
) -> pd.DataFrame:
    """Generate transaction data based on customer behavior profiles."""
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range_days = (end - start).days
    
    # Product categories and their characteristics
    categories = {
        'Electronics': {'price_range': (50, 1500), 'weight': 0.20},
        'Fashion': {'price_range': (20, 300), 'weight': 0.25},
        'Home & Garden': {'price_range': (15, 500), 'weight': 0.15},
        'Beauty': {'price_range': (10, 150), 'weight': 0.15},
        'Sports': {'price_range': (25, 400), 'weight': 0.10},
        'Books': {'price_range': (10, 50), 'weight': 0.10},
        'Toys': {'price_range': (15, 200), 'weight': 0.05}
    }
    
    # Profile-based transaction parameters
    profile_params = {
        'vip': {'freq_range': (15, 40), 'value_mult': 1.8, 'recency_max': 30},
        'loyal': {'freq_range': (10, 25), 'value_mult': 1.3, 'recency_max': 60},
        'regular': {'freq_range': (5, 15), 'value_mult': 1.0, 'recency_max': 120},
        'occasional': {'freq_range': (2, 8), 'value_mult': 0.8, 'recency_max': 200},
        'dormant': {'freq_range': (1, 4), 'value_mult': 0.6, 'recency_max': 400},
        'new': {'freq_range': (1, 5), 'value_mult': 1.0, 'recency_max': 45}
    }
    
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Bank Transfer']
    
    transactions = []
    transaction_id = 1
    
    for _, customer in customers_df.iterrows():
        profile = customer['behavior_profile']
        params = profile_params[profile]
        
        # Determine number of transactions for this customer
        n_transactions = np.random.randint(params['freq_range'][0], params['freq_range'][1] + 1)
        
        # Generate transactions spread over time based on profile
        if profile == 'new':
            # New customers have transactions only in recent period
            cust_start = end - timedelta(days=90)
        elif profile == 'dormant':
            # Dormant customers stopped buying recently
            cust_start = start
            most_recent = end - timedelta(days=params['recency_max'])
        else:
            cust_start = max(start, datetime.combine(customer['join_date'], datetime.min.time()))
        
        for _ in range(n_transactions):
            # Generate transaction date
            if profile == 'dormant':
                txn_date = fake.date_time_between(start_date=cust_start, end_date=most_recent)
            elif profile == 'new':
                txn_date = fake.date_time_between(start_date=cust_start, end_date=end)
            else:
                # Bias towards more recent transactions for active profiles
                days_ago = int(np.random.exponential(scale=params['recency_max']))
                days_ago = min(days_ago, (end - cust_start).days)
                txn_date = end - timedelta(days=days_ago)
                if txn_date < cust_start:
                    txn_date = fake.date_time_between(start_date=cust_start, end_date=end)
            
            # Select category
            cat_names = list(categories.keys())
            cat_weights = [categories[c]['weight'] for c in cat_names]
            category = np.random.choice(cat_names, p=cat_weights)
            
            # Generate price based on category and profile
            price_range = categories[category]['price_range']
            base_price = np.random.uniform(price_range[0], price_range[1])
            unit_price = round(base_price * params['value_mult'], 2)
            
            # Quantity (VIPs tend to buy more)
            if profile == 'vip':
                quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])
            else:
                quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            
            total_amount = round(unit_price * quantity, 2)
            
            transactions.append({
                'transaction_id': f'TXN_{transaction_id:06d}',
                'customer_id': customer['customer_id'],
                'transaction_date': txn_date.strftime('%Y-%m-%d'),
                'product_category': category,
                'product_name': f"{category} Item",
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': total_amount,
                'payment_method': np.random.choice(payment_methods),
                'country': customer['country']
            })
            transaction_id += 1
    
    return pd.DataFrame(transactions)


def generate_full_dataset(
    n_customers: int = 5000,
    save_path: str = None
) -> tuple:
    """Generate complete e-commerce dataset."""
    
    print("Generating customer base...")
    customers_df = generate_customers(n_customers)
    
    print("Generating transactions...")
    transactions_df = generate_transactions(customers_df)
    
    print(f"Generated {len(customers_df)} customers")
    print(f"Generated {len(transactions_df)} transactions")
    
    if save_path:
        transactions_df.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")
    
    return customers_df, transactions_df


if __name__ == "__main__":
    customers, transactions = generate_full_dataset(
        n_customers=5000,
        save_path="data/raw/ecommerce_transactions.csv"
    )
    print("\nSample transactions:")
    print(transactions.head(10))
