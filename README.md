# Reconstructing Inter-Regional Transmission and Material Bridges in Asian Buddhist Epigraphy and Portable Icons

Computational geometry and Topological Data Analysis (TDA) pipeline accompanying the manuscript submitted to *npj Heritage Science* (Submission ID: `b6cfe0f6-524d-4990-8518-5874c38cb945`).

## Repository Structure

- `code/01_corpus_ingestion.py`: TEI-XML & Museum API ingestion and harmonization
- `code/02_persistent_homology.py`: Multi-modal TF-IDF/SVD metric space & Ripser TDA (H0, H1)
- `code/03_network_bridges.py`: 1-skeleton betweenness centrality & cross-corpus bridges
- `code/requirements.txt`: Python package dependencies
- `data/raw_data/`: 308 DHARMA XML inscriptions, 100 V&A JSON records, 100 CMA JSON records
- `data/processed_data/`: Harmonized CSV corpus and pairwise distance matrix
- `figures/`: High-resolution 300 DPI manuscript figures (`Fig1`–`Fig3`)

## Reproducibility Instructions

1. Install dependencies: `pip install -r code/requirements.txt`
2. Run corpus harmonization: `cd code && python 01_corpus_ingestion.py`
3. Run persistent homology & Figures 1–2: `python 02_persistent_homology.py`
4. Run network centrality & Figure 3: `python 03_network_bridges.py`
