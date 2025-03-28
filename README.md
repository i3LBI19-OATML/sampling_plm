# Sampling Proteins

Consists of 2 parts:
1. Protein Generation 
2. Evaluation Framework

## Protein Generation

Generation Steps[^3]:
1. Score possible mutated/extended sequence using Tranception
2. Generate mutations key
3. Sample mutations/extension
4. Mutate/extend original sequence with key
5. Repeat (1-4) until desired parameters

[^3]: Mutate for MLDE and extend for AR

### Mutation/Extension with PLM
Might be transferable to any PLM

| Files | Description |
| --- | --- |
| `generator.py` | IRS-singles |
| `multi-generator.py` | IRS-doubles |
| `AR_generator.py` | ARS-singles & ARS-doubles|

### Leveraging `.generate()` from HuggingFace
Transferable to other autoregressive HuggingFace model. Generates from left to right direction.

| Files | Description |
| --- | --- |
| `auto-gen.py` | Autoregressive generation using prompts |
| `auto-mask-gen.py` | Masks locations on the sequence and fill the mask autoregressively |

## Protein Evaluation

Calculation for various sequence- and structure-based quality scores for proteins, such as those produced by generative models.

| Files | Description |
| --- | --- |
| `protein_scoring.py`[^1] [^2] | Evaluates protein sequences from fasta files |

Metrics:
1. Structue metrics
2. Single-sequence metrics
3. Alignment-based metrics

[^1]: Requires MSA reference (DMS_msa_files.zip) and weights (DMS_msa_weights.zip), such as those found in ProteinGym.

[^2]: Requires reference in fasta format. If using ProteinGym MSA files as reference, changing file extension .a2m to .fasta should be enough.

## Supplementary Information
### Preparing EVmutation
```bash
EVmutation/plmc/bin/plmc -o [model output name].model_params \ 
                         -c [model output name].txt \
                         -f [name of target sequence (WT) from MSA file, before forwardslash (/)] \
                         -le [0.2*(N-1), N is length of target sequence] \
                         -lh 0.01 \
                         -m 200 \
                         -t 0.2 \
                         -g [MSA file (.a2m format)]
```
*MSA files downloadable from [here](https://github.com/OATML-Markslab/Tranception/tree/main?tab=readme-ov-file#multiple-sequence-alignments-msas)

<!--
## Reference
If you use this repository in your work, please cite the following paper:
```
Darmawan, J. T., Gal, Y., & Notin, P. (2023). Sampling protein language models for functional protein design. In NeurIPS 2023 Generative AI and Biology (GenBio) Workshop.
```

### BibTeX
```bibtex
@inproceedings{
darmawan2023sampling,
title={Sampling Protein Language Models for Functional Protein Design},
author={Jeremie Theddy Darmawan and Yarin Gal and Pascal Notin},
booktitle={NeurIPS 2023 Generative AI and Biology (GenBio) Workshop},
year={2023},
url={https://openreview.net/forum?id=JPOW9FToYX}
}
```

## Links
- NeurIPS 2023 GenBio proceedings: https://openreview.net/pdf?id=JPOW9FToYX
- NeurIPS 2023 MLSB proceedings: https://www.mlsb.io/papers_2023/Sampling_Protein_Language_Models_for_Functional_Protein_Design.pdf
-->
