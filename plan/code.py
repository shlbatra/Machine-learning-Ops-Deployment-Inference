# from duckdb.experimental.spark.sql import SparkSession
import pandas as pd
import numpy as np
from faker import Faker
import uuid
import datetime

fake = Faker()

def main() -> None:
    NUM_USERS_TO_GENERATE = 500
    NUM_TRANSFERS_TO_GENERATE = 2000
#     spark = SparkSession.builder.getOrCreate()
    users_dataframe = generate_user_data(NUM_USERS_TO_GENERATE)
    transfers_dataframe = generate_transfer_data(NUM_TRANSFERS_TO_GENERATE, users_dataframe)

def generate_user_data(num_users: int) -> pd.DataFrame:
    print(f"Generating {num_users} user records...")
    COUNTRIES = ['GB', 'US', 'DE', 'AU', 'SG', 'BE', 'FR', 'CA']
    COUNTRY_PROBS = [0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05]
    ACCOUNT_TYPES = ['Personal', 'Business']
    ACCOUNT_TYPE_PROBS = [0.85, 0.15] # 85% Personal, 15% Business
    KYC_STATUSES = ['Verified', 'Pending', 'Rejected']
    KYC_STATUS_PROBS = [0.9, 0.07, 0.03] # Most users are verified
    users_data = []
    for _ in range(num_users):
        user = {
            'user_id': str(uuid.uuid4()),
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'email': fake.unique.email(),
            'country_of_residence': np.random.choice(COUNTRIES, p=COUNTRY_PROBS),
            'account_type': np.random.choice(ACCOUNT_TYPES, p=ACCOUNT_TYPE_PROBS),
            'kyc_status': np.random.choice(KYC_STATUSES, p=KYC_STATUS_PROBS),
            'registration_date': fake.date_time_between(start_date='-3y', end_date='now', tzinfo=datetime.timezone.utc),
            'label': np.random.choice([0, 1], p=[0.9, 0.1])
        }
        users_data.append(user)
    return pd.DataFrame(users_data)


def generate_transfer_data(num_transfers: int, users_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nGenerating {num_transfers} transfer records...")
    CURRENCIES = ['GBP', 'EUR', 'USD', 'AUD', 'JPY', 'CAD']
    TRANSFER_STATUSES = ['COMPLETED', 'PENDING', 'CANCELLED', 'FAILED']
    TRANSFER_STATUS_PROBS = [0.92, 0.03, 0.03, 0.02] # Most transfers complete
    exchange_rate = np.random.normal(loc=1.15, scale=0.05)
    user_ids = users_df['user_id'].tolist()
    transfers_data = []
    for _ in range(num_transfers):
        sender_id, recipient_id = np.random.choice(user_ids, size=2, replace=False)
        source_amount = np.random.lognormal(mean=np.log(100), sigma=1.5)
        transfer = {
            'transfer_id': str(uuid.uuid4()),
            'sender_id': sender_id,
            'recipient_id': recipient_id,
            'source_currency': np.random.choice(CURRENCIES),
            'target_currency': np.random.choice(CURRENCIES),
            'source_amount': round(source_amount, 2),
            'target_amount': round(source_amount * exchange_rate, 2),
            'status': np.random.choice(TRANSFER_STATUSES, p=TRANSFER_STATUS_PROBS),
            'created_at': fake.date_time_between(start_date='-2y', end_date='now', tzinfo=datetime.timezone.utc)
        }
        transfers_data.append(transfer)
    return pd.DataFrame(transfers_data)


if __name__ == "__main__":
    main()