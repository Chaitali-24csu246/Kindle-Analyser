import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Kindle Dataset Analyzer", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    return df

def clean_dataframe(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    expected = ["asin", "title", "author", "soldBy", "imgUrl", "productURL",
                "stars", "reviews", "price", "isKindleUnlimited", "year", "genre"]

    col_map = {}
    for col in df.columns:
        low = col.lower()
        for e in expected:
            if low == e.lower():
                col_map[col] = e
                break
    df = df.rename(columns=col_map)

    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    def parse_stars(x):
        try:
            if pd.isna(x):
                return np.nan
            s = str(x).split()[0].replace(",", "")
            return float(s)
        except:
            return np.nan

    df["stars"] = df["stars"].apply(parse_stars)

    def parse_int(x):
        try:
            if pd.isna(x):
                return 0
            s = "".join(ch for ch in str(x) if ch.isdigit() or ch == "-")
            return int(s) if s != "" else 0
        except:
            return 0

    df["reviews"] = df["reviews"].apply(parse_int)

    def parse_price(x):
        try:
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            if s.lower() in ["free", "0", "0.0", "0.00"]:
                return 0.0
            for ch in ["$", "₹", "Rs.", "Rs", "USD", "INR"]:
                s = s.replace(ch, "")
            if "-" in s:
                parts = [p.strip() for p in s.split("-") if p.strip() != ""]
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p.replace(",", "")))
                    except:
                        pass
                if len(nums) == 0:
                    return np.nan
                return float(np.mean(nums))
            s = s.replace(",", "")
            return float(s)
        except:
            return np.nan

    df["price"] = df["price"].apply(parse_price)

    def parse_bool(x):
        if pd.isna(x):
            return False
        s = str(x).strip().lower()
        if s in ["true", "t", "yes", "y", "1", "ku"]:
            return True
        if s in ["false", "f", "no", "n", "0"]:
            return False
        return "kindleunlimited" in s or "kindle unlimited" in s

    df["isKindleUnlimited"] = df["isKindleUnlimited"].apply(parse_bool)
    df["author"] = df["author"].fillna("Unknown Author").astype(str)
    df["title"] = df["title"].fillna("Unknown Title").astype(str)
    df["reviews_per_book"] = df["reviews"]
    df["stars_filled"] = df["stars"].fillna(df["stars"].median())

    return df

# ---------- UI ----------
st.title("Kindle Dataset Analyzer")
st.markdown("Helps you analyze your dataset of Kindle e-books, authors, ratings, and trends.")

with st.sidebar:
    st.header("Upload & Options")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    st.caption("Expected columns: asin, title, author, stars, reviews, price, isKindleUnlimited")
    st.markdown("---")
    top_n = st.slider("Top N for leaderboards", 5, 50, 10)
    show_images = st.checkbox("Show cover thumbnails in tables", value=True)
    st.markdown("---")
    st.write("Export cleaned data:")
    export_clean = st.button("Prepare cleaned CSV for download")

if uploaded_file is not None:
    raw = load_csv(uploaded_file)
    st.success(f"Loaded dataset with {raw.shape[0]:,} rows and {raw.shape[1]} columns.")

    df = clean_dataframe(raw)

    # Quick summary numbers
    st.markdown("---")
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Unique authors", f"{df['author'].nunique():,}")
    col3.metric("Avg Rating (mean)", f"{df['stars'].mean():.2f}")
    col4.metric("Avg Reviews (mean)", f"{df['reviews'].mean():.0f}")

    # Filters
    st.markdown("---")
    st.subheader("Filters")
    mask = pd.Series(True, index=df.index)

    sel_author = st.text_input("Search author by name")
    if sel_author.strip() != "":
        mask = mask & df["author"].str.contains(sel_author, case=False)

    ku_filter = st.selectbox("Kindle Unlimited filter", options=["All", "Only KU", "Only non-KU"], index=0)
    if ku_filter == "Only KU":
        mask = mask & df["isKindleUnlimited"]
    elif ku_filter == "Only non-KU":
        mask = mask & (~df["isKindleUnlimited"])

    min_reviews = st.number_input("Min reviews (filter)", min_value=0, value=0, step=1)
    if min_reviews > 0:
        mask = mask & (df["reviews"] >= min_reviews)

    df_filtered = df[mask].copy()
    st.write(f"Filtered records: {len(df_filtered):,}")

    if st.checkbox("View filtered dataset"):
        st.dataframe(df_filtered)

    # ---------- Tabs ----------
    tab_overview, tab_authors, tab_ratings, tab_price, tab_ku, tab_export, tab_predict, tab_qa = st.tabs(
        ["Overview", "Authors", "Ratings", "Price", "Kindle Unlimited", "Export", "Predictive Analysis", "Q&A"]
    )

    # ---------- Overview tab ----------
    with tab_overview:
        st.subheader("Basic distributions")
        sns.set(style="whitegrid", palette="pastel")
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        sns.histplot(df_filtered["stars"].dropna(), kde=False, bins=20, ax=axes[0])
        axes[0].set_title("Ratings distribution (stars)")
        sns.histplot(df_filtered["reviews"], bins=30, ax=axes[1], log_scale=(False, True))
        axes[1].set_title("Reviews distribution (log scale y)")
        sns.histplot(df_filtered["price"].dropna(), bins=30, ax=axes[2])
        axes[2].set_title("Price distribution")
        st.pyplot(fig)

        st.markdown("### Top titles by reviews")
        DEFAULT_IMG = "https://via.placeholder.com/80x120.png?text=No+Cover"
        top_titles = df_filtered.sort_values("reviews", ascending=False).head(top_n)[
            ["title", "author", "stars", "reviews", "price", "isKindleUnlimited", "imgUrl"]
        ]
        for _, row in top_titles.iterrows():
            cols = st.columns([1, 4])
            img_url = row["imgUrl"] if pd.notna(row["imgUrl"]) else DEFAULT_IMG
            cols[0].image(img_url, width=60)
            cols[1].markdown(f"**{row['title']}** by {row['author']}  \nStars: {row['stars']}, Reviews: {row['reviews']}, Price: {row['price']}, KU: {row['isKindleUnlimited']}")

    # ---------- Authors tab ----------
    with tab_authors:
        st.subheader("Top authors by number of books")
        authors_count = df.groupby("author").agg(
            books_count=("asin", "nunique"),
            avg_stars=("stars", "mean"),
            total_reviews=("reviews", "sum")
        ).reset_index().sort_values("books_count", ascending=False)
        st.dataframe(authors_count.head(top_n))

        fig, ax = plt.subplots(figsize=(10, 5))
        top_authors_plot = authors_count.head(top_n).sort_values("books_count", ascending=True)
        ax.barh(top_authors_plot["author"], top_authors_plot["books_count"], color=sns.color_palette("pastel"))
        ax.set_xlabel("Number of books")
        ax.set_title(f"Top {top_n} authors by number of books")
        st.pyplot(fig)

    # ---------- Ratings tab ----------
    with tab_ratings:
        st.subheader("Ratings analysis")
        st.write(df_filtered["stars"].describe())

    # ---------- Price tab ----------
    with tab_price:
        st.subheader("Price analysis")
        st.write(df_filtered["price"].describe())

    # ---------- Kindle Unlimited tab ----------
    with tab_ku:
        st.subheader("Kindle Unlimited (KU) vs Non-KU comparison")
        ku_group = df_filtered.groupby("isKindleUnlimited").agg(
            count=("asin", "nunique"),
            avg_stars=("stars", "mean"),
            avg_price=("price", "mean"),
            total_reviews=("reviews", "sum")
        ).reset_index()
        ku_group["isKindleUnlimited"] = ku_group["isKindleUnlimited"].map({True: "Kindle Unlimited", False: "Non-KU"})
        st.dataframe(ku_group)

    # ---------- Export tab ----------
    with tab_export:
        st.subheader("Export cleaned dataset (sample or full)")
        if export_clean:
            to_export = df.copy()
            cols = ["asin", "title", "author", "stars", "reviews", "price", "isKindleUnlimited", "productURL", "imgUrl"]
            to_export = to_export[cols]
            csv = to_export.to_csv(index=False).encode("utf-8")
            st.download_button("Download cleaned CSV", data=csv, file_name="kindle_cleaned.csv", mime="text/csv")
        else:
            st.write("Click the button in the sidebar to prepare cleaned CSV for download.")

    # ---------- Predictive Analysis tab ----------
    with tab_predict:
        st.subheader("Predictive Analysis")
        st.write("Placeholder for predictive analytics like estimating ratings or reviews based on book features.")

    # ---------- Q&A tab ----------
    with tab_qa:
        st.subheader("Ask a question")
        user_question = st.text_input("Enter your question here")
        if user_question:
            q = user_question.lower()
            if "top author" in q and "2020" in q:
                if "year" in df.columns:
                    temp = df[df["year"]==2020]
                    if temp.empty:
                        st.write("Insufficient data")
                    else:
                        st.write(f"Top author in 2020: {temp['author'].value_counts().idxmax()}")
                else:
                    st.write("Insufficient data")
            elif "most popular genre" in q:
                if "genre" in df.columns and df["genre"].notna().any():
                    st.write(f"Most popular genre: {df['genre'].value_counts().idxmax()}")
                else:
                    st.write("Insufficient data")
            else:
                st.write("Question not recognized or data unavailable.")

else:
    st.info("Please upload the Kindle CSV dataset to begin analysis.")
