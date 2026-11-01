"""Toys, Toys, Toys Inc.: customer feedback and owner analytics."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from labeler import CATEGORIES, label_comment

# ---- Settings ----------------------------------------------------------
TITLE = "Toys, Toys, Toys Inc."
LOGO = Path(__file__).with_name("logo.png")
CSV_FILE = Path(__file__).with_name("feedback.csv")
LOW_RATING, HIGH_RATING = 2, 4
COLUMNS = [
    "submitted_at", "comment", "rating", "sentiment", "category",
]
COLORS = {
    "price": "#577590",
    "quality": "#43AA8B",
    "variety": "#9C64A6",
    "delivery": "#F9C74F",
    "service": "#F3722C",
}
SENTIMENT_COLORS = {
    "positive": "#2F9E61",
    "negative": "#D9534F",
    "unclear": "#999999",
}
# ------------------------------------------------------------------------

st.set_page_config(page_title=TITLE, page_icon=str(LOGO), layout="wide")
st.image(str(LOGO), width=115)
st.title(TITLE)
st.caption("Customers submit toy feedback. An LLM classifies it for owner analysis.")


def load_feedback():
    if not CSV_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    data = pd.read_csv(CSV_FILE, keep_default_na=False)
    for column in COLUMNS:
        if column not in data.columns:
            data[column] = ""
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    return data[COLUMNS]


def save_feedback(data, new_row):
    pd.concat([data, pd.DataFrame([new_row])], ignore_index=True).to_csv(
        CSV_FILE, index=False
    )


def manual_review_reason(row):
    reasons = []
    if row["sentiment"] not in ("positive", "negative"):
        reasons.append("Unclear sentiment")
    if row["category"] not in CATEGORIES:
        reasons.append("Unclear category")
    if row["sentiment"] == "positive" and row["rating"] <= LOW_RATING:
        reasons.append("Positive comment with low rating")
    if row["sentiment"] == "negative" and row["rating"] >= HIGH_RATING:
        reasons.append("Negative comment with high rating")
    return "; ".join(reasons)


df = load_feedback()
tab_customer, tab_owner = st.tabs(
    ["Customer: leave feedback", "Owner: analyze feedback"]
)

# ---- Customer side -----------------------------------------------------
with tab_customer:
    st.subheader("Tell us about your toy purchase")
    with st.form("feedback_form", clear_on_submit=True):
        comment = st.text_area("Your comment", placeholder="What did you like or dislike?")
        stars = st.feedback("stars")  # 0-4 internally; stored as 1-5
        submitted = st.form_submit_button("Send feedback")

    if submitted:
        if not comment.strip() or stars is None:
            st.error("Both a written comment and a star rating are required.")
        else:
            try:
                labels = label_comment(comment)
                save_feedback(df, {
                    "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "comment": comment.strip(),
                    "rating": stars + 1,
                    **labels,
                })
                df = load_feedback()
            except Exception as exc:
                st.error(f"Feedback could not be saved: {exc}")
            else:
                st.success("Thank you for your feedback!")
                st.caption(
                    f"Sentiment: {labels['sentiment']} | "
                    f"Category: {labels['category']}"
                )

# ---- Owner side --------------------------------------------------------
with tab_owner:
    st.subheader("What is the one area that needs immediate attention?")
    st.caption("Category ratings reflect the primary topic in each customer comment.")

    if df.empty:
        st.info("No feedback yet. Submit feedback in the Customer tab.")
    else:
        df = df.copy()
        df["review_reason"] = df.apply(manual_review_reason, axis=1)
        needs_review = df[df["review_reason"] != ""]
        known_categories = df[df["category"].isin(CATEGORIES)]
        ratings = known_categories.groupby("category")["rating"].agg(["mean", "count"])
        ratings = ratings.sort_values(["mean", "category"], ascending=[False, True])

        # Four summary metrics: overall rating, highest category, lowest
        # category, and comments requiring manual review.
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average star rating", f"{df['rating'].mean():.2f} / 5")

        if not ratings.empty:
            best_category = ratings.index[0]
            worst_category = ratings.index[-1]
            m2.metric("Highest rated category", best_category.title())
            m2.caption(
                f"{ratings.loc[best_category, 'mean']:.2f} stars "
                f"(n={int(ratings.loc[best_category, 'count'])})"
            )
            m3.metric("Lowest rated category", worst_category.title())
            m3.caption(
                f"{ratings.loc[worst_category, 'mean']:.2f} stars "
                f"(n={int(ratings.loc[worst_category, 'count'])})"
            )
            if len(ratings) == 1:
                m3.caption("Only one category has classified feedback so far.")
        else:
            m2.metric("Highest rated category", "N/A")
            m3.metric("Lowest rated category", "N/A")

        m4.metric("Comments needing manual review", len(needs_review))

        # Exactly two graphs, grouped by the same primary feedback category.
        chart1, chart2 = st.columns(2)
        # Count each primary category by sentiment. Include unclear sentiment so
        # the total height of each bar matches the number of categorized comments.
        # A missing category is not plotted, but is flagged for manual review.
        sentiment_order = ["positive", "negative", "unclear"]
        chart_data = known_categories.copy()
        chart_data["sentiment"] = chart_data["sentiment"].where(
            chart_data["sentiment"].isin(sentiment_order), "unclear"
        )
        chart_data = (
            chart_data.groupby(["category", "sentiment"]).size()
            .reindex(
                pd.MultiIndex.from_product(
                    [CATEGORIES, sentiment_order], names=["category", "sentiment"]
                ),
                fill_value=0,
            )
            .rename("count")
            .reset_index()
        )
        chart_data["bar_label"] = chart_data["count"].where(
            chart_data["count"] > 0, ""
        )

        with chart1:
            st.markdown("**Number of comments by category and sentiment**")
            fig = px.bar(
                chart_data,
                x="category", y="count", color="sentiment",
                text="bar_label", barmode="stack",
                color_discrete_map=SENTIMENT_COLORS,
                category_orders={
                    "category": list(CATEGORIES),
                    "sentiment": sentiment_order,
                },
                labels={"sentiment": "Sentiment"},
            )
            fig.update_layout(
                xaxis_title=None, yaxis_title="Comments", legend_title_text="Sentiment",
            )
            fig.update_traces(textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green: positive; red: negative; grey: unclear sentiment.")

        with chart2:
            st.markdown("**Average stars by label**")
            avg_data = known_categories.groupby("category", as_index=False)["rating"].mean()
            avg_data["category"] = pd.Categorical(
                avg_data["category"], categories=CATEGORIES, ordered=True
            )
            avg_data = avg_data.sort_values("category")
            if avg_data.empty:
                st.info("No categorized feedback to graph yet.")
            else:
                fig = px.bar(
                    avg_data, x="category", y="rating", color="category",
                    text_auto=".2f", color_discrete_map=COLORS, range_y=[0, 5],
                )
                fig.update_layout(
                    showlegend=False, xaxis_title=None, yaxis_title="Average stars",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Comments requiring manual review")
        st.caption(
            "Flags include unclear classifications or conflicting star ratings "
            "and comment sentiment (1-2 stars with "
            "positive text, or 4-5 stars with negative text)."
        )
        if needs_review.empty:
            st.success("No comments currently require manual review.")
        else:
            # Render the complete comment as text; dataframe cells can truncate it.
            for _, row in needs_review.iterrows():
                with st.container(border=True):
                    st.write(row["comment"])
                    st.caption(
                        f"Stars: {int(row['rating']) if pd.notna(row['rating']) else 'N/A'} | "
                        f"Sentiment: {row['sentiment']} | Category: {row['category']} | "
                        f"Review status: Needs manual review"
                    )
                    st.caption(f"Reason for review: {row['review_reason']}")

        with st.expander("All feedback"):
            st.dataframe(df.drop(columns=["review_reason"]), hide_index=True,
                         use_container_width=True)
        st.download_button(
            "Download feedback CSV",
            df.drop(columns=["review_reason"]).to_csv(index=False),
            file_name="toy_feedback.csv", mime="text/csv",
        )
        st.caption(
            "CSV storage is temporary on some hosted deployments, including free "
            "Streamlit Cloud. Download the data or use persistent storage for production."
        )
