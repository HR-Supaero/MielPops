from pathlib import Path
import pandas as pd
import plotly.express as px

from imbalance_reasonable import count_images_per_class, TRAIN_DIR

def plot_class_distribution(df):

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
        title="Nombre d'images par espèce",
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
    df_top = df.head(top_n)

    fig = px.pie(
        df_top.reset_index(),
        names="class_name",
        values="n_images",
        title=f"Top {top_n} classes",
        template="plotly_white"
    )

    fig.show()


if __name__ == "__main__":
    df_counts = count_images_per_class(TRAIN_DIR)
    print(df_counts.head())

    plot_class_distribution(df_counts)
    # plot_histogram(df_counts)
    # plot_pie(df_counts, top_n=len(df_counts))

    print("Nombre total d'images :", df_counts["n_images"].sum())
    print("Nombre de classes :", len(df_counts))
    print("Classe la plus représentée :", df_counts.index[0])
    print("Classe la moins représentée :", df_counts.index[-1])