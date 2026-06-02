import random
import os
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Ensure NLTK data is present
nltk.download('brown', quiet=True)

print("--- Booting Evaluation Engine ---")
print("Fetching 10,000 sentences from the Brown Corpus...")

# 1. Generate the Real vs. Fake dataset
num_sentences = 5000
real_sentences = [' '.join(sent).lower() for sent in brown.sents()[:num_sentences]]
words = [word.lower() for word in brown.words()[:num_sentences * 10] if word.isalpha()]
fake_sentences = [' '.join(random.sample(words, k=random.randint(5, 10))) for _ in range(num_sentences)]

X_text = real_sentences + fake_sentences
y = [1] * len(real_sentences) + [0] * len(fake_sentences) 

# 2. Vectorize using 2 to 3-grams
print("Vectorizing Text...")
vectorizer = CountVectorizer(ngram_range=(2, 3))
X = vectorizer.fit_transform(X_text)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training Logistic Regression Model...\n")
classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
classifier.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = classifier.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Fake/Unnatural (0)', 'Real/Natural (1)'])
cm = confusion_matrix(y_test, y_pred)

print("="*50)
print("MODEL EVALUATION: LOGISTIC REGRESSION")
print("="*50)
print(report)

print("Confusion Matrix:")
print(f"True Negatives (Correctly spotted fake): {cm[0][0]}")
print(f"False Positives (Thought a fake was real): {cm[0][1]}")
print(f"False Negatives (Thought a real was fake): {cm[1][0]}")
print(f"True Positives (Correctly spotted real): {cm[1][1]}")
print("="*50)

# --- GRAPH GENERATION SECTION ---
print("\nGenerating visual reports for slides...")

# A. Save Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Fake', 'Predicted Real'],
            yticklabels=['Actual Fake', 'Actual Real'])
plt.title('Confusion Matrix: Context Scorer Performance')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved: confusion_matrix.png")

# B. Save Bar Chart of Key Metrics
# Extracting Precision, Recall, and F1 for the "Real" class (index 1)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
accuracy = (y_pred == y_test).mean()

metric_labels = ['Accuracy', 'Precision (Real)', 'Recall (Real)', 'F1-Score (Real)']
metric_values = [accuracy, precision[1], recall[1], f1[1]]

plt.figure(figsize=(10, 6))
sns.barplot(x=metric_labels, y=metric_values, palette='viridis')
plt.ylim(0, 1.0)
plt.title('Key Model Performance Metrics')
plt.ylabel('Score (0.0 - 1.0)')

# Add text labels on top of bars
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_metrics.png')
print("Saved: model_metrics.png")

print("\nEvaluation Complete. You can now pull the .png files into your PPT.")