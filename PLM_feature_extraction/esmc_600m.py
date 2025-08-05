import torch
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        current_label = None
        current_sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_label:
                    sequences[current_label] = ''.join(current_sequence)
                current_label = line[1:] 
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_label:
            sequences[current_label] = ''.join(current_sequence)
    return sequences

file_path = './data/train/XU_pretrain_total.txt'

sequences = read_fasta(file_path)

model_name = 'esmc_600m'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ESMC.from_pretrained(model_name).to(device)

embeddings = []

for label, sequence in sequences.items():
    protein = ESMProtein(sequence=sequence)

    protein_tensor = model.encode(protein).to(device)

    with torch.no_grad():
        logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

    sequence_embedding = torch.mean(logits_output.embeddings, dim=1).squeeze().cpu().numpy()

    embeddings.append([label] + sequence_embedding.tolist())

df = pd.DataFrame(embeddings)

df.columns = ['name'] + [f'embedding_{i}' for i in range(1, df.shape[1])]
df['target'] = df['name'].apply(lambda x: 1 if 'positive' in x else 0)

csv_file_path = './feature/train/esmc_600m.csv'
df.to_csv(csv_file_path, index=False)

print(f"Embedding results saved to {csv_file_path}")
