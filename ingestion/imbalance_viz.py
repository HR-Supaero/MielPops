from pathlib import Path
import pandas as pd
import plotly.express as px
import os

from imbalance_reasonable import count_images_per_class, TRAIN_DIR

def plot_class_distribution(df, title="Nombre d'images par espèce"):

    df_plot = df.reset_index()

    # couleurs alternées
    colors = [
        "gold" if i % 2 == 0 else "black"
        for i in range(len(df_plot))
    ]

    fig = px.bar(
        df_plot,
        x="class_name",
        y="n_images",
        title=title,
        template="plotly_white"
    )

    fig.update_traces(marker_color=colors)

    fig.update_layout(
        xaxis_title="Espèce",
        yaxis_title="Nombre d'images",
        xaxis_tickangle=-45,
    )

    fig.show()

def plot_histogram(df):
    fig = px.histogram(
        df,
        x="n_images",
        nbins=30,
        title="Distribution du nombre d'images par classe",
        template="plotly_white"
    )

    fig.show()

def plot_pie(df, top_n=10):
    df = df.sort_values("n_images", ascending=False)
    df_top = df.head(top_n)

    other = pd.DataFrame({
        "index": ["Other"],
        "n_images": [df.iloc[top_n:]["n_images"].sum()]
    }).set_index("index")

    df_final = pd.concat([df_top, other])
    print(other)

    fig = px.pie(
        df_final.reset_index(),
        names="index",
        values="n_images",
        title=f"Top {top_n} classes",
        template="plotly_white"
    )
    fig.update_traces(pull=[0.1 if i=="Other" else 0 for i in df_final.index])

    fig.show()


if __name__ == "__main__":
    df_counts = count_images_per_class(TRAIN_DIR)
    # print(df_counts)

    plot_class_distribution(df_counts, f"Nombre d'images par espèce dans le dataset original {TRAIN_DIR}")
    # plot_histogram(df_counts)
    # plot_pie(df_counts, top_n=21)

    print("Nombre total d'images :", df_counts["n_images"].sum())
    print("Nombre de classes :", len(df_counts))
    print("Classe la plus représentée :", df_counts.index[0])
    print("Classe la moins représentée :", df_counts.index[-1])

    treated_image_path = "./data/treated_train/"
    if os.path.exists(treated_image_path):

        df_counts_treated = count_images_per_class(treated_image_path)
        plot_class_distribution(df_counts_treated, f"Nombre d'images par espèce dans le dataset augmenté {treated_image_path}")
        # print(df_counts_treated)