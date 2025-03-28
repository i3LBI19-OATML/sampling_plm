# import os
import pandas as pd

def save_as_fasta(df: pd.DataFrame, fasta_output):
    # Open the output file for writing
    with open(fasta_output, 'a') as f:
        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            # Write the header line (starting with ">")
            f.write('>' + row['name'] + '\n')
            # Write the sequence data (wrapped at 80 characters per line)
            sequence = row['sequence']
            while len(sequence) > 0:
                f.write(sequence[:80] + '\n')
                sequence = sequence[80:]