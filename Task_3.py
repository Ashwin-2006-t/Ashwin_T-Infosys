# 🧩 Task 3 – Feature Correlation and Text Insights

# 🎯 Objective:
# Explore how non-text features like has_company_logo, telecommuting, and employment_type relate to fraudulent vs. real jobs,
# and visualize language differences using WordClouds.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ✅ Step 1: Import data
df = pd.read_csv('fake_job_postings (1).csv')

# Ensure description text is clean
df['clean_description'] = df['description'].fillna('').str.lower()

# ✅ Step 2: Select relevant metadata columns
cols = ['has_company_logo', 'telecommuting', 'employment_type', 'required_experience', 'fraudulent']
meta_df = df[cols]
print(meta_df.head())

# ✅ Step 3: Compare proportions using grouped counts

# 🔹 A. Company Logo Presence
logo_stats = df.groupby('fraudulent')['has_company_logo'].value_counts(normalize=True).unstack()
logo_stats.plot(kind='bar', stacked=False, color=['skyblue', 'salmon'])
plt.title('Company Logo Presence vs Fraudulent Jobs')
plt.xlabel('Fraudulent (0 = Real, 1 = Fake)')
plt.ylabel('Proportion')
plt.legend(['No Logo (0)', 'Has Logo (1)'])
plt.show()

# 🔹 B. Remote Work (Telecommuting)
tele_stats = df.groupby('fraudulent')['telecommuting'].value_counts(normalize=True).unstack()
tele_stats.plot(kind='bar', color=['lightgreen', 'orange'])
plt.title('Remote Work (Telecommuting) vs Fraudulent Jobs')
plt.xlabel('Fraudulent (0 = Real, 1 = Fake)')
plt.ylabel('Proportion')
plt.legend(['Office Job (0)', 'Remote (1)'])
plt.show()

# 🔹 C. Employment Type
emp_stats = df.groupby(['fraudulent', 'employment_type']).size().unstack(fill_value=0)
(emp_stats.div(emp_stats.sum(axis=1), axis=0)
 .T.plot(kind='bar', figsize=(8,4), color=['lightblue', 'lightcoral']))

plt.title('Employment Type Distribution: Real vs Fake')
plt.xlabel('Employment Type')
plt.ylabel('Proportion')
plt.legend(['Real (0)', 'Fake (1)'])
plt.show()

# ✅ Step 4: Create WordClouds for Real vs Fake Descriptions
real_text = ' '.join(df[df['fraudulent'] == 0]['clean_description'].dropna())
fake_text = ' '.join(df[df['fraudulent'] == 1]['clean_description'].dropna())

real_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_text)
fake_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(fake_text)

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.imshow(real_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Real Job Descriptions', fontsize=16)

plt.subplot(1,2,2)
plt.imshow(fake_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Fake Job Descriptions', fontsize=16)
plt.show()
