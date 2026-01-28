# Kindle Analyzer
. Basic analysis program on kindle dataset
. Matplotlib, Seaborn, Numpy, Pandas, Streamlit

An interactive Streamlit-based data analysis app to explore Kindle e-book datasets.
It helps analyze authors, ratings, reviews, prices, Kindle Unlimited trends, and more — all through a clean web UI.
Built using Python, Pandas, Seaborn, Matplotlib, and Streamlit.

. Features


Upload any Kindle CSV dataset, or one provided in repo


Automatic data cleaning & normalization


Interactive filters (author, reviews, Kindle Unlimited)


Visual insights:


Ratings distribution


Reviews distribution (log scale)


Price trends




Author leaderboards


Kindle Unlimited vs Non-KU comparison


Export cleaned dataset as CSV


Simple Q&A interface for dataset queries(placeholder)



. Tech Stack


Python 3.9+


Streamlit


Pandas & NumPy


Matplotlib & Seaborn

.How to Run This Project Locally


1️. Clone the repository
git clone https://github.com/Chaitali-24csu246/Kindle-Analyser.git
cd Kindle-Analyser


2️. (Optional) Create a Virtual Environment
Avoids dependency issues 
macOS / Linux
python3 -m venv venv
source venv/bin/activate

Windows
python -m venv venv
venv\Scripts\activate


3️. Install Dependencies
pip install streamlit pandas numpy matplotlib seaborn


4️. Run the Streamlit App
streamlit run kindle_dashboard.py


5️. Open in Browser
Streamlit will automatically open a browser tab.
If not, visit:
http://localhost:8501


Note: press ctrl+c to exit terminal


. How to Use
Note(Some features are just placeholders)

Upload your Kindle CSV dataset


Apply filters (author name, review count, KU status)


Explore tabs:


Overview


Authors


Ratings


Price


Kindle Unlimited




Download the cleaned dataset if needed



 Exporting Cleaned Data


Click “Prepare cleaned CSV for download” in the sidebar


Download a normalized CSV file 
