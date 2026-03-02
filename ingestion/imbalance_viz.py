from pathlib import Path
import pandas as pd
import plotly.express as px

TRAIN_DIR = "data/train"

def count_images_per_class(train_dir: str, extension: str = ".jpg") -> pd.DataFrame:
    """
    Parcourt un dossier train et compte le nombre d'images par sous-dossier.

    Parameters
    ----------
    train_dir : str
        Chemin vers data/train
    extension : str
        Extension des fichiers image à compter

    Returns
    -------
    pd.DataFrame
        DataFrame indexée par nom de classe avec colonne 'n_images'
    """

    train_path = Path(train_dir)

    data = []

    # Parcours des sous-dossiers (espèces)
    for subfolder in sorted(train_path.iterdir()):
        if subfolder.is_dir():
            n_images = len(list(subfolder.glob(f"*{extension}")))

            data.append({
                "class_name": subfolder.name,
                "n_images": n_images
            })

    # Création dataframe
    df = pd.DataFrame(data)
    df = df.set_index("class_name").sort_values("n_images", ascending=False)

    return df

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