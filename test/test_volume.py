import pandas as pd

df = pd.read_csv('data/crypto_raw.csv')
btc = df[df['Symbol']=='BTC']
eth = df[df['Symbol']=='ETH']
bnb = df[df['Symbol']=='BNB']

print('BTC Volume statistics:')
print(f'  Non-null: {btc["Volume"].notna().sum()}/{len(btc)}')
print(f'  Mean: {btc["Volume"].mean()}')
print(f'  Max: {btc["Volume"].max()}')
print(f'  Min: {btc["Volume"].min()}')
print(f'  Sample values: {btc["Volume"].head().tolist()}')

print('\nETH Volume statistics:')
print(f'  Non-null: {eth["Volume"].notna().sum()}/{len(eth)}')
print(f'  Mean: {eth["Volume"].mean()}')
print(f'  Max: {eth["Volume"].max()}')
print(f'  Min: {eth["Volume"].min()}')

print('\nBNB Volume statistics:')
print(f'  Non-null: {bnb["Volume"].notna().sum()}/{len(bnb)}')
print(f'  Mean: {bnb["Volume"].mean()}')
print(f'  Max: {bnb["Volume"].max()}')
print(f'  Min: {bnb["Volume"].min()}')