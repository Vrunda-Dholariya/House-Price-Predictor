# 🏠 California House Price Predictor

A live Machine Learning web app that predicts the median house price of a
California district based on its features — built with a Random Forest
regression model and deployed with Streamlit.

**🔗 Live Demo:** _(paste your Streamlit link here after deploying)_

---

## 📌 What it does
Enter details about a housing district (median income, house age, average rooms,
location, etc.) and the app predicts the **median house price** in real time.

## 🧠 Machine Learning concept
This is a **regression** problem — the model predicts a continuous number
(a price), unlike classification which predicts a category.

## 🛠️ Tech Stack
- **Python**
- **Pandas** — data handling
- **Scikit-learn** — Random Forest regression model
- **Streamlit** — web interface and deployment

## ⚙️ How it works
1. Loads the built-in California Housing dataset (20,640 districts, 8 features).
2. Splits the data into 80% training / 20% testing.
3. Trains a `RandomForestRegressor` to learn the relationship between the
   features and the house price.
4. Takes user input from sliders and predicts the price live.

## 📊 Results
| Model | R² Score | Avg. Error |
|-------|----------|------------|
| Linear Regression | 0.58 | ~$53,000 |
| **Random Forest** | **0.81** | **~$33,000** |

Switching from Linear Regression to Random Forest improved the R² score from
**0.58 → 0.81** by capturing non-linear patterns in the data.

## ▶️ Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---
Built by **Vrunda Dholariya** as part of my AI/ML learning journey. 🚀
