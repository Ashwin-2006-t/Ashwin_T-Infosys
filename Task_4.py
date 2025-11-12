# 🧩 Task 4 – Keyword-Based Fake Job Detector (Rule-Based Prototype)

# 🎯 Objective:
# Build a simple rule-based fake job detector using suspicious keywords in job descriptions.

import pandas as pd

# ✅ Step 1: Load dataset
df = pd.read_csv('fake_job_postings (1).csv')

# Ensure text column is ready
df['clean_description'] = df['description'].fillna('').str.lower()

# ✅ Step 2: Define suspicious keywords
suspicious_keywords = [
    'urgent', 'work from home', 'limited vacancy', 'visa',
    'investment', 'training fee', 'money transfer', 'payment', 'registration fee'
]

# ✅ Step 3: Define rule-based function
def rule_based_flag(text):
    text = str(text).lower()
    for word in suspicious_keywords:
        if word in text:
            return 1   # suspect job
    return 0          # normal job

# ✅ Step 4: Apply rule to dataset
df['suspect_flag'] = df['clean_description'].apply(rule_based_flag)

# ✅ Step 5: Check overlap between predicted suspect jobs and actual fraudulent labels
cross_tab = pd.crosstab(df['suspect_flag'], df['fraudulent'], normalize='all')
print("\n📊 Overlap between Rule-based Flag and True Fraudulent Labels:\n")
print(cross_tab)

# ✅ Step 6: Show examples of suspect but real jobs
false_positives = df[(df['suspect_flag'] == 1) & (df['fraudulent'] == 0)]
print("\n🔍 Examples of Suspect but Real Jobs:\n")
print(false_positives[['title', 'clean_description']].head(5))
