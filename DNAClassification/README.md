
# DNA Classification

## [Dataset](https://www.kaggle.com/datasets/miadul/dna-classification-dataset)

## Introduction

This dataset contains 3,000 synthetic DNA samples with 13 features designed for genomic data analysis, machine learning, and bioinformatics research. Each row represents a unique DNA sample with both sequence-level and statistical attributes.

## Column Names

Each entry of the dataset has information about the following columns:

| Column Name       | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `Sample_ID`       | Unique identifier for each DNA sample.                       |
| `Sequence`        | DNA sequence (string of A, T, C, G).                         |
| `GC_Content`      | Percentage of Guanine (G) and Cytosine (C) in the sequence.  |
| `AT_Content`      | Percentage of Adenine (A) and Thymine (T) in the sequence.   |
| `Sequence_Length` | Total sequence length.                                       |
| `Num_A`           | Number of Adenine bases.                                     |
| `Num_T`           | Number of Thymine bases.                                     |
| `Num_C`           | Number of Cytosine bases.                                    |
| `Num_G`           | Number of Guanine bases.                                     |
| `kmer_3_freq`     | Average 3-mer (triplet) frequency score                      |
| `Mutation_Flag`   | Binary flag indicating mutation presence (0 = No, 1 = Yes).  |
| `Class_Label`     | Class of the sample (Human, Bacteria, Virus, Plant).         |
| `Disease_Risk`    | Risk level associated with the sample (Low / Medium / High). |

## Conclusions

If we could get real data, this could be a very interesting topic to do researchs. Classification algorithms could work really good on them, of course not perfect as the models on the notebook. Some Deep Learning approaches could be applied to datasets like this and find out excellent conclusions.
