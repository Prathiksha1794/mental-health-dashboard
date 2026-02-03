# MSc Data Science Dissertation
This repository contains the code, methodology, and documentation for my MSc Data Science dissertation.

## Data Access
Due to size constraints, raw datasets are not included in this GitHub repository.
The full dataset used in this project is available here: https://drive.google.com/drive/folders/1TwZesJwEo66zULYdt0VRf0M1fMw37_97

**AI-Augmented Early Intervention System for Mental Health Detection on Social Media
Project Overview**
Welcome to the repository for the MSc Data Science Research Project, a comprehensive study leveraging natural language processing (NLP) and machine learning to analyse mental health discourse on Reddit. This project classifies posts into six DSM-5 categories—anxiety, autism, bipolar disorder, borderline personality disorder (BPD), depression, and schizophrenia—while addressing challenges such as class imbalance, semantic overlap, and model interpretability. The workflow integrates exploratory data analysis, advanced feature engineering, robust model training, and explainability techniques, offering a reproducible pipeline for mental health research. Ethical considerations are central, with raw data restricted to protect user privacy, and only anonymised datasets are provided.

Project Structure
Index of Files
Code Files
EDA_and_SHAP.ipynb: Loads and cleans the Reddit mental health dataset (titles + text combined).Performs preprocessing steps including lowercasing, punctuation removal, stopword elimination, and lemmatisation.Explores the dataset through visualisations such as Class Distribution, Word frequency analysis, etc. Trains a baseline classifier, applies SHAP to identify important words driving classification decisions.

Traditional_models.ipynb: In this code file, we load the dataset. Applies TF-IDF vectorisation to transform text into numerical feature vectors. Trains and tests a range of traditional classifiers: Logistic Regression, Decision Tree, Gradient Boosting, etc. The purpose of this code file is to establish baseline machine learning performance benchmarks and highlight class-level performance differences.

binary_classification.ipynb: This notebook conducts binary classification experiments to simplify mental health text classification, comparing performance against multi-class settings. It filters the cleaned dataset into binary categories (e.g., depression vs. non-depression), applies TF-IDF vectorisation, and encodes labels for supervised learning. Baseline classifiers (Logistic Regression, Decision Tree, SVM) are trained and evaluated using accuracy, precision, recall, and F1-score. Results show higher accuracy in binary classification due to reduced complexity, but highlight challenges in multi-class settings due to semantic overlaps (e.g., depression vs. anxiety). This analysis underscores the potential of targeted binary classifiers for clinical applications while emphasising the need for robust multi-class models aligned with DSM-5 categories.

SVM, LIME with SVM & LR.ipynb: In this notebook file, we develop and evaluate a Support Vector Machine (SVM) classifier, complemented with LIME (Local Interpretable Model-Agnostic Explanations) for interpretability. Logistic Regression is also implemented for comparison.

SVm+LIME without MH .ipynb: This notebook is the same as the previous file, but in this one, the SVM and the LIME were done and compared without the mental health class.

word2vec.ipynb: This notebook leverages Word2Vec to create semantic vector representations of Reddit posts, capturing contextual relationships beyond frequency-based methods. It preprocesses text through cleaning, tokenisation, and stopword removal, then trains a Gensim Word2Vec model to generate dense word embeddings. These are aggregated into document vectors for classification, revealing insights like the proximity of “panic” to “attack.”

Rulefit_final.ipynb: This notebook applies the RuleFit algorithm to classify Reddit posts into mental health categories, focusing on interpretability through human-readable decision rules.. It preprocesses text using TF-IDF, encodes subreddit labels, and applies RuleFit to generate and weight rules from decision tree ensembles. The model’s performance is evaluated via accuracy, precision, recall, and F1-score, compared against baselines like Logistic Regression and SVM.Key rules (e.g., “IF panic AND attack THEN Anxiety”) provide transparent insights into linguistic drivers of predictions, bridging accuracy and explainability in mental health classification.

bert_project: This folder consists of all the files related BERT model, In BERT model we have done a BERT-based text classification model for categorising Reddit mental health posts into subreddits (e.g., depression, anxiety, bipolar, autism). It leverages the pre-trained BERT tokeniser to convert text into contextualised vector representations and fine-tunes a BERT model with a classification head. The model is trained on a preprocessed dataset (with and without the generic mental health class) and evaluated using accuracy, F1-scores, and classification reports, benchmarked against traditional models (SVM, RuleFit, Word2Vec). By capturing deep semantic relationships, BERT outperforms TF-IDF-based approaches, effectively handling overlaps in subreddit categories.

Lexicon.ipynb: This notebook extracts lexicon-based features to capture the emotional tone of Reddit posts, enhancing statistical and deep learning models with interpretable psychological insights. It processes cleaned, tokenised text and applies the NRC Emotion Lexicon to map words to eight emotions (anger, anticipation, disgust, fear, joy, sadness, surprise, trust) and sentiment polarities. Emotion scores are normalised and compared across subreddit categories, revealing patterns like elevated sadness in depression and fear in anxiety. These features complement TF-IDF and embeddings, bridging linguistic and psychological insights for transparent classification.

LDA Topic Modelling.ipynb: This notebook employs Latent Dirichlet Allocation (LDA) to uncover latent themes in Reddit mental health posts, offering an unsupervised lens on discourse. It preprocesses text by tokenising, removing stopwords, and lemmatising, then constructs a dictionary and corpus for Gensim-based LDA. The model, trained with 7 topics aligned to subreddit classes, extracts key keywords and assigns psychological labels (e.g., psychosis, emotional struggle). Interactive pyLDAvis visualisations and topic distribution plots reveal thematic overlaps, with depression/anxiety focusing on hopelessness and fatigue, bipolar on mood swings and medication, and autism on social challenges and identity.

mental_health_test.py: There is a Streamlit app (MindScope) that loads a fine-tuned BERT model to classify free-text into mental-health categories. It displays the predicted label with confidence and a probability chart.

**Datasets of the Project**
Mental-health-related-subreddits: The raw dataset containing posts from all the subreddits used for initial EDA and baseline experiments.
Mental_health_preprocesseddataset: The dataset saved after EDA and preprocessing steps, like removing the unwanted and small classes with fewer than 2 or 3 posts.
Mental-health-without MH-subreddits: The refined dataset with the mentalhealth class removed and preprocessed as well. Used for model training and evaluation to address label ambiguity and improve classifier performance.
**Visualisations
Tableau Visualisation**
It is a type of Dashboard where Visualisations and analytical results, featuring SHAP explanations for decision tree classifiers (with and without the mental health class), confusion matrices for binary (e.g., mental health vs. depression) and multi-class tasks, top 30 terms from RuleFit models (by importance and absolute count), emotional tone distributions, LDA topic modelling with 7 topics, and LIME interpretability for SVM and Logistic Regression are present. It offers a comprehensive summary of model performance, interpretability, and thematic insights across subreddit categories.
