from transformers import T5Tokenizer, T5EncoderModel
import torch
import pandas as pd

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd")

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

embeddings = []

for label, sequence in sequences.items():
    sequence = ' '.join(sequence)

    inputs = tokenizer(sequence, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state

    sequence_embedding = torch.mean(embedding, dim=1).squeeze().numpy()

    embeddings.append([label] + sequence_embedding.tolist())

df = pd.DataFrame(embeddings)

df.columns = ['name'] + [f'embedding_{i}' for i in range(1, df.shape[1])]
df['target'] = df['name'].apply(lambda x: 1 if 'positive' in x else 0)

csv_file_path = './feature/train/prot_t5_xl_bfd.csv'
df.to_csv(csv_file_path, index=False)

print(f"Embedding results saved to {csv_file_path}")
