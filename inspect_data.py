import pandas as pd

try:
    df = pd.read_excel('code/fyp dataset.xlsx', sheet_name='DetailedTrustDataset')
    print("Columns:", df.columns.tolist())
    print("First row:", df.iloc[0].to_dict())
    print("Shape:", df.shape)
except Exception as e:
    print(e)
