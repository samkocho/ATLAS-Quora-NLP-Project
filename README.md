# Quora Question Pairs - NLP Project

This repository contains the code and analysis for the Quora Question Pairs competition on Kaggle. The goal of the competition is to identify which questions asked on Quora are duplicates of each other. This can be useful to provide better answers to users by merging duplicate questions and improving the overall question-answering experience.

## Project Structure

The project is divided into several Jupyter notebooks, each focusing on different aspects of the analysis:

### Notebooks

1. **Exploratory Data Analysis (EDA) (`QuoraEDA.ipynb`)**
   - **1. EDA**: 
     - **Data Overview**: Summary statistics and initial examination of the dataset.
     - **Distribution Analysis**: Visualization of the distribution of questions and pair counts.
     - **Missing Values**: Analysis and handling of missing values in the dataset.
     - **Text Analysis**: Word frequency analysis, common words, and n-gram analysis.
     - **Duplicate Analysis**: Examination of duplicate question pairs and their characteristics.

2. **Feature Engineering and Modeling (`QuoraFeatureEngineering&Modeling.ipynb`)**
   - **1. Preprocess Data**:
     - Removing non-word characters using regular expressions.
     - Converting text to lowercase for uniformity.
     - Splitting the text into tokens (words).
   - **2. Feature Engineering**:
     - **TF-IDF**: Implementing Term Frequency-Inverse Document Frequency for text representation.
     - **Word Embeddings**: Using pre-trained word vectors for feature extraction.
     - **Similarity Measures**: Calculating cosine similarity and other distance metrics.
   - **3. Modeling**:
     - **Model Selection**: Evaluating different machine learning models (e.g., Logistic Regression, SVM, Random Forest).
     - **Hyperparameter Tuning**: Using Grid Search and Cross-Validation to optimize model performance.
     - **Model Evaluation**: Analyzing model performance using metrics such as accuracy, precision, recall, and F1-score.
   - **4. Advanced Techniques**:
     - **Sentence Embeddings**: Creating numerical representations of the questions using advanced techniques like BERT or Universal Sentence Encoder.
     - **Neural Networks**: Implementing deep learning models for improved performance.

## Getting Started

### Prerequisites

To run the notebooks and reproduce the analysis, you will need the following packages:

- Python 3.x
- Jupyter Notebook
- Pandas
- Numpy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- TensorFlow (for sentence embeddings)
- Transformers (for BERT and other advanced models)

## Results

The results of the analysis, including the performance metrics of various models, can be found in the respective notebooks. Detailed explanations and visualizations are provided to aid understanding.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome improvements and new ideas.

## Acknowledgments

Kaggle for providing the dataset and hosting the competition.
The open-source community for the tools and libraries used in this project



