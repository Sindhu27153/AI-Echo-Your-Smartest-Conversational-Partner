
# 🤖 AI Echo: Your Smartest Conversational Partner

AI Echo is an intelligent NLP-based sentiment analysis system designed to understand user emotions from ChatGPT reviews. This project leverages machine learning and deep learning models to classify user sentiments into **Positive**, **Neutral**, or **Negative**, offering valuable insights for product teams, researchers, and developers.

---

## 📌 1. Problem Statement

Sentiment analysis helps determine the emotional tone behind textual data. In this project, we aim to analyze user reviews of a ChatGPT application and classify them based on sentiment. The goal is to:

- Understand customer satisfaction
- Identify common concerns
- Enhance overall user experience

---

## 💼 2. Business Use Cases

- ✅ **Customer Feedback Analysis** – Improve product features by understanding user sentiments.
- ✅ **Brand Reputation Management** – Monitor perception trends over time.
- ✅ **Feature Enhancement** – Pinpoint improvements from negative/neutral reviews.
- ✅ **Automated Customer Support** – Prioritize and classify reviews for quick response.
- ✅ **Marketing Strategy Optimization** – Target promotions based on sentiment clusters.

---

## 🔍 3. Approach

### 📂 Data Preprocessing

- Cleaning text: Removing punctuation, stopwords, special characters
- Tokenization & Lemmatization
- Handling missing values
- Language detection (if required)
- Text normalization (lowercasing, trimming)

### 📊 Exploratory Data Analysis (EDA)

- Sentiment distribution analysis
- Word cloud visualization
- Time series trends
- Review length vs sentiment
- Location and platform-wise sentiment comparison

### ⚙️ Model Development

- Feature Extraction: TF-IDF Vectorization
- Model Training:
  - Naïve Bayes
  - Logistic Regression
  - Random Forest
  - Support Vector Classifier (SVC)
  - LSTM  (optional for enhancement)
- Model Selection based on evaluation metrics

### 📈 Evaluation Metrics

- Accuracy
- Precision / Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score and Curve

---

## 🚀 4. Results

- 🔍 **Sentiment Distribution:** Positive, Negative, and Neutral breakdown
- 🧠 **Top Performing Model:** Support Vector Classifier with class balancing
- 📊 **Insights:**
  - Negative sentiments tied to performance and UI
  - Verified users show stronger sentiment polarities
  - Mobile platforms have more neutral reviews
- 🔧 **Recommendations:**
  - Focus on speed and clarity for UI updates
  - Improve mobile responsiveness
  - Track version-wise issues for better QA

---

## 🧪 5. Technologies Used

- **Languages & Frameworks:** Python, Streamlit
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Seaborn, Matplotlib, WordCloud
- **Deployment:** Streamlit (with optional AWS or Heroku)

---

## 📦 Folder Structure

AI-Echo-Sentiment-Analysis/ 
├── 📁 data/ # Raw and processed data 
├── 📁 notebooks/ # Jupyter notebooks for EDA and model training
├── 📁 models/ # Pickle files for trained models & vectorizers 
├── 📁 app/ # Streamlit web app 
├── 📁 reports/ # Evaluation metrics and visualization
├── README.md 
└── requirements.txt

## 🎯 How to Run the App

1. Clone the repo:
   ```bash
   git clone https://github.com/sindhu27152/ai-echo-sentiment-analysis.git
   cd ai-echo-sentiment-analysis

2.Install dependencies:
  
  pip install -r requirements.txt

3.Launch the Streamlit app:
  
  streamlit run app/app.py

📜 License
This project is licensed under the MIT License.

📞 Contact
Developed by: Sindhuja Seenivasan
Email: sindhujaakp@gmail.com
LinkedIn: https://www.linkedin.com/in/sindhuja-s-a3974593/





