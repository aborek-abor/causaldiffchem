import re

print('Cleaning zinc250k.csv...')
with open('zinc250k.csv', 'r') as f:
    content = f.read()

smiles_list = re.findall(r'"([A-Za-z0-9@\[\]\(\)\+\-=#\\\/.%:\n]+?)"', content)
smiles_clean = []
for s in smiles_list:
    s = s.replace('\n', '').strip()
    if len(s) > 5:
        smiles_clean.append(s)

with open('zinc250k_clean.smi', 'w') as f:
    for s in smiles_clean:
        f.write(s + '\n')

print(f'Done. Extracted {len(smiles_clean):,} SMILES to zinc250k_clean.smi')
