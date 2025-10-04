# SpaceX Falcon 9 First Stage Landing Prediction

![Falcon 9 Landing](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/landing_1.gif)

This is an end-to-end Data Science Capstone Project from IBM's Coursera course. It predicts whether the Falcon 9 first stage will land successfully, enabling cost estimation for launches (~$62M reusable vs. ~$165M expendable). The pipeline covers data collection (API & web scraping), wrangling, EDA, visualization, SQL analysis, and machine learning modeling.

Key goal: Use historical data to train models that forecast landing success, highlighting factors like orbit type, payload mass, and launch site.

## Project Structure

The project is organized into Jupyter notebooks following a data science workflow:

- **01_Data_Collection_API_Raw.ipynb**: Fetches raw launch data from the SpaceX API (rockets, payloads, etc.). Outputs `data/dataset_part_1.csv` (~100 launches up to 2021).
- **02_Data_Collection_WebScraping.ipynb**: Scrapes Wikipedia for supplemental historical records using BeautifulSoup. Outputs `data/spacex_web_scraped.csv` (~106 rows).
- **03_Data_Wrangling_Cleaning.ipynb**: Cleans, merges, and engineers features (e.g., binary `Class` label: 1=success, 0=failure). Outputs `data/dataset_part_2.csv`.
- **04_EDA_SQL_Analysis.ipynb**: Loads data to SQLite; runs SQL queries for insights (e.g., failure rankings by date range).
- **05_EDA_DataViz.ipynb**: Exploratory analysis with Seaborn/Matplotlib (e.g., success by orbit/year). One-hot encodes features. Outputs `data/dataset_part_3.csv` (ML-ready).
- **06_Machine_Learning_Prediction.ipynb**: Builds pipelines with Scikit-learn (Logistic Regression, SVM, etc.); evaluates with GridSearchCV (best: ~83% accuracy).

## Key Findings & Insights

- **Success Rate**: ~67% overall; rises to ~95% for LEO orbits and post-2015 (drone ship improvements).
- **Trends**: Early failures (2010-2014: 0% success); heavy payloads (>10k kg) riskier for GTO/HEO.
- **Top Factors**: Launch site (CCAFS safest), reuse count, grid fins presence.
- **Model Performance**: Logistic Regression (83% test accuracy) > SVM/Tree/KNN; low false positives minimize cost risks.
- **SQL Highlights**: Most common outcome: "No attempt" (10 in 2010-2017); avg. payload ~5k kg from CCSFS.

Visual Example (from 05_EDA_DataViz.ipynb):
- Bar chart: LEO highest success (~95%), VLEO ~80%.
- Line trend: Success jumps from 0% (2010) to 100% (2020).

## Tech Stack

- **Languages/Tools**: Python 3.12, Jupyter Notebook/Lab.
- **Data Handling**: Pandas, NumPy, Requests, BeautifulSoup.
- **Viz**: Matplotlib, Seaborn.
- **DB**: SQLAlchemy, SQLite.
- **ML**: Scikit-learn (GridSearchCV, pipelines).
- **Other**: No external installs beyond basics (see requirements.txt).

## Setup & Running

1. **Clone the Repo**:
   ```
   git clone https://github.com/Haroon-89/SpaceX-Landing-Predictor.git
   cd SpaceX-Landing-Predictor
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   (Or in Jupyter: `%pip install pandas numpy scikit-learn matplotlib seaborn beautifulsoup4 sqlalchemy requests`.)

3. **Run Notebooks**:
   - Open in Jupyter: `jupyter lab` (or `jupyter notebook`).
   - Execute sequentially: Start with 01 → 06.
   - CSVs auto-generate in `/data/`; use them for quick jumps.

4. **Regenerate Data**:
   - API calls in 01 may need API keys (public endpoints work).
   - Scraping in 02 uses static Wikipedia snapshot—update URL for live data.

## Datasets

- **dataset_part_1.csv**: Raw (~90 rows, 17 cols: FlightNumber, Date, PayloadMass, etc.).
- **dataset_part_2.csv**: Processed (~90 rows, + `Class` label).
- **dataset_part_3.csv**: Numeric features (~90 rows, 8 cols: ready for modeling).
- **spacex_web_scraped.csv**: Scraped supplement (~106 rows, 11 cols).

Data is historical (up to 2021); for 2025 updates (350+ launches, >98% success), re-run notebooks or extend API calls.

## Results & Model Evaluation

- **Best Model**: Logistic Regression (Test Accuracy: 83.33%).
- Confusion Matrix (sample): High precision on successes (reuse predictions).
- Future Work: Add XGBoost; deploy via Streamlit; fetch real-time data.

| Model                  | CV Score | Test Accuracy |
|------------------------|----------|---------------|
| Logistic Regression   | 0.85    | 0.8333       |
| SVM                    | 0.82    | 0.8000       |
| Decision Tree         | 0.78    | 0.7667       |
| KNN                    | 0.80    | 0.8000       |

## License

MIT License © 2025 [Your Name]. Feel free to fork/extend!

## Acknowledgments

- IBM Skills Network / Coursera: "Applied Data Science Capstone".
- SpaceX API & Wikipedia: Data sources.
- Authors: Joseph Santarcangelo, Nayef Abou Tayoun, Yan Luo, Pratiksha Verma.

Questions? Open an issue or [contact me](mailto:your.email@example.com). Let's iterate on reusability! 🚀
