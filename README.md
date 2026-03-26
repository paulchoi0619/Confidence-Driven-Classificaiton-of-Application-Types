# Confidence-Driven Classification of Application Types

This repository contains experiments for **application type classification from network traffic** using deep learning and confidence-driven analysis methods. The project explores multiple experimental settings, including session-based classification, filtered traffic classification, 10-class classification, and GMM-based clustering of learned embeddings.

## Repository Structure

The repository is organized into the following main experiment folders:

### `sesson_based_label_experiment`
Session-based experiment where traffic is labeled according to the session from which it originated.

### `filtered_traffic_experiment`
This experiment removes background traffic and focuses only on classifying application traffic.

### `softmax_10_classes_experiment`
A 10-class classification experiment in which background traffic is treated as a separate category.

### `gmm_clustering_experiment`
Contains notebooks for clustering cosine similarity score vector using Gaussian Mixture Models (GMM).

## Models

The repository includes Jupyter notebooks for the following models:

- **BiLSTM**
- **FS-Net**

These models are used in different experiment settings for traffic classification.

## Setup
1. Download `requirements.txt`.
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
## Dataset
The datasets for this project can be downloaded from the **Releases** section of this repository.
This project provides two versions of the dataset in CSV file:

### Domain-labeled dataset
Includes domain name along with human-annotated application categories.
### Labeled dataset
Includes only the application category labels.

## Preprocessing and CSV Parsing
The code for parsing PCAP files and reading CSV files was inspired by the **FlowPic** project:
https://github.com/talshapira/FlowPic

## Experiments
### Session-Based Experiment
Labels traffic based on the session in which it originated.
### Filtered Traffic Experiment
Removes background traffic and classifies only application traffic.
### 10-Class Experiment
Classifies traffic into 10 classes, where background traffic is included as a separate class.
### GMM Clustering
Includes scripts for clustering learned embeddings and cosine similarity scores.

## Notes
- Most of the implementation is provided in **Jupyter notebooks**.
- This repository is intended for experimentation and analysis of network traffic classification methods.
- Download both dataset versions and place them in the **main project directory** so they are located at the same level as the experiment folders.
