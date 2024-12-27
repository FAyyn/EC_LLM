import pandas as pd
from collections import defaultdict, Counter
import math
import nltk
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm

# Set NLTK_DATA environment variable
nltk_data_dir = '/usr/share/nltk_data'
os.environ['NLTK_DATA'] = nltk_data_dir

if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download nltk data
try:
    print("Downloading NLTK data packages...")
    nltk.download('punkt', download_dir=nltk_data_dir)
    print("NLTK data packages downloaded successfully.")
except Exception as e:
    print(f"Error downloading punkt: {e}")
    exit(1)

def calculate_tf_idf(texts):
    print("Calculating TF-IDF...")
    n_grams_all = defaultdict(Counter)
    total_texts = len(texts)

    # Progress bar for processing texts
    for text_idx, text in enumerate(tqdm(texts, desc="Processing texts")):
        words = word_tokenize(text.lower())
        for i in range(1, 5):
            for j in range(len(words) - i + 1):
                n_gram = ' '.join(words[j:j+i])
                n_grams_all[i][n_gram] += 1

    print("Counting document frequencies...")
    doc_freq = {n: defaultdict(int) for n in n_grams_all}
    
    # Count document frequencies
    for text in tqdm(texts, desc="Counting document frequencies"):
        unique_n_grams = set()
        words = word_tokenize(text.lower())
        for i in range(1, 5):
            for j in range(len(words) - i + 1):
                n_gram = ' '.join(words[j:j+i])
                if n_gram not in unique_n_grams:
                    doc_freq[i][n_gram] += 1
                    unique_n_grams.add(n_gram)

    print("Calculating TF-IDF scores...")
    tf_idf = defaultdict(float)
    
    # Calculate TF-IDF scores
    for n in n_grams_all:
        total_n_grams = sum(n_grams_all[n].values())
        for n_gram, count in n_grams_all[n].items():
            tf = count / total_n_grams
            idf = math.log(total_texts / (1 + doc_freq[n][n_gram]))
            tf_idf[n_gram] = tf * idf

    print("TF-IDF calculation completed.")
    return tf_idf

def cider_score(candidate, reference, tf_idf):
    candidate_words = word_tokenize(candidate.lower())
    reference_words = word_tokenize(reference.lower())
    scores = []
    
    for i in range(1, 5):
        candidate_n_grams = { ' '.join(candidate_words[j:j+i]) for j in range(len(candidate_words) - i + 1) }
        reference_n_grams = { ' '.join(reference_words[j:j+i]) for j in range(len(reference_words) - i + 1) }
        common_n_grams = candidate_n_grams.intersection(reference_n_grams)
        
        score = sum(tf_idf.get(n_gram, 0) for n_gram in common_n_grams)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0

# Read Excel file
print("Reading Excel file...")
df = pd.read_excel('/workspace/DataProcess/测试问题（带微调后答案）.xlsx', engine='openpyxl')
print("Excel file read successfully.")

# Assume columns
standard_answers = df['答案'].tolist()
original_answers = df['llama3回答'].tolist()
fine_tuned_answers = df['微调后回答'].tolist()

# Filter non-strings
standard_answers = [str(ans) for ans in standard_answers if isinstance(ans, str) or isinstance(ans, float)]
original_answers = [str(ans) for ans in original_answers if isinstance(ans, str) or isinstance(ans, float)]
fine_tuned_answers = [str(ans) for ans in fine_tuned_answers if isinstance(ans, str) or isinstance(ans, float)]

# Calculate TF-IDF
print("Calculating TF-IDF...")
tf_idf = calculate_tf_idf(standard_answers + original_answers + fine_tuned_answers)
print("TF-IDF calculated.")

# Calculate CIDEr scores
print("Calculating CIDEr scores for original and fine-tuned models...")
df['CIDEr_Original'] = [cider_score(original, standard, tf_idf) for original, standard in zip(original_answers, standard_answers)]
df['CIDEr_FineTuned'] = [cider_score(fine_tuned, standard, tf_idf) for fine_tuned, standard in zip(fine_tuned_answers, standard_answers)]
print("CIDEr scores calculated successfully.")

# Write results to Excel
print("Writing results to Excel file...")
df.to_excel('/workspace/DataProcess/CIDEr1.xlsx', index=False, engine='openpyxl')
print("Results written to Excel file successfully.")
