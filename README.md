# Football Performance and Team Success

> Data Visualization - Final Project - Escola Superior de Gestão e Tecnologia - Politechnic Institute of Santarém

Interactive Dashboard in **Streamlit** Which Analyzes the Performance of European Soccer Teams Using Expected Goals (xG) Data from [Understat](https://understat.com/). It Includes Exploratory Analysis, Correlations, Team-level and Season-level Aggregations, and Machine Learning Models for Predicting Match Outcomes.

---

## Project Structure

```
football-performance-team-success/
├── data/
│   ├── understat_per_game.csv      # Principal Dataset (per game)
│   └── understat.com.csv           # Complementary Dataset
├── dashboard.py                    # Streamlit Dashboard
├── main.ipynb                      # Original Notebook
├── requirements.txt                # Python Dependencies
├── run.bat                         # Execution Script (Windows)
├── run.sh                          # Execution Script (Linux/macOS)
└── README.md
```

---

## How-To Run

### Option 1 - Automatic Script

**Windows** - Double-click in run.bat File or Run in the Terminal:

```bat
run.bat
```

**Linux Based System:**

```bash
chmod +x run.sh
./run.sh
```

> The Scripts Automatically Install Dependencies and Start the Dashboard.

### Option 2 - Manual

1. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

2. **Start the Dashboard:**

```bash
streamlit run dashboard.py
```

3. Open the Browser in **http://localhost:8501** and Enjoy!

---

## Dependencies

| Package | Usage |
|---|---|
| `streamlit` | Interactive Dashboard Framework |
| `pandas` | Data Manipulation and Analysis |
| `numpy` | Numerical Operations |
| `plotly` | Interactive Graphics |
| `scikit-learn` | Machine Learning Models |
| `statsmodels` | Scatter Plots |

---

## Dashboard Features

| Tab | Content |
|---|---|
| **Overview** | Dataset Summary, Descriptive Statistics and Missing Values |
| **EDA** | Categorical and Numerical Variables vs Points |
| **Correlations** | Correlation Matrix, Distributions and Heatmap of Top Features |
| **Teams / Seasons** | Club Rankings, xG vs Points Scatter, Best Seasons |
| **Machine Learning** | Logistic Regression and Random Forest - Confusion Matrices, ROC Curves, Feature Importance |
| **Variables** | Data Dictionary with Explanation of Each Variable |

---

## Notes

- The `understat_per_game.csv` Dataset is Loaded Automatically From the `data/` Folder, but can be Removed in Top Left of the Sidebar.
- All Loading Functions use **cache** (`@st.cache_data`) to Avoid Re-processing and Re-load Full Dataset Again.
- The Filters in the Sidebar Allow you to Explore Subsets by League, Season, and Team.
