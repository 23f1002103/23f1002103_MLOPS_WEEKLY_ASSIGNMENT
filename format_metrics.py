import csv

with open('metrics.csv') as f:
    row = next(csv.DictReader(f))

print('| Metric | Value |')
print('|---|---|')
for k, v in row.items():
    print(f'| {k} | {float(v):.4f} |')
