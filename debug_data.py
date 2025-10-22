import pandas as pd

# Read first few rows
df = pd.read_csv('./data/Compound_000000001_000025000_props_geom.csv', nrows=10)

print("Columns:", df.columns.tolist())
print("\n" + "="*60)

# Check bonding_status
print("\nChecking bonding_status column:")
for i in range(min(5, len(df))):
    val = df['bonding_status'].iloc[i]
    print(f"  Row {i}: {repr(val)} (type: {type(val).__name__})")

print("\n" + "="*60)

# Check bonding_issues_atoms
print("\nChecking bonding_issues_atoms column:")
for i in range(min(5, len(df))):
    val = df['bonding_issues_atoms'].iloc[i]
    print(f"  Row {i}: {repr(val)} (type: {type(val).__name__})")
    if isinstance(val, str):
        print(f"    Length: {len(val)}")
        print(f"    Is it '[]'? {val == '[]'}")

print("\n" + "="*60)

# Count bonding_status values
print("\nBonding status distribution:")
print(df['bonding_status'].value_counts())
