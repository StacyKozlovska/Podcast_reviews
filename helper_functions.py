"""
Helper functions necessary for the data analysis of Podcast Reviews.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
from wordcloud import WordCloud
from varname import nameof
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import process
from pprint import pprint
import textwrap


sns.set_style("darkgrid")

xticklabels_d = {
    "horizontalalignment": "right",
    "fontweight": "light",
    "fontsize": "x-large",
}

def plot_sns_barplot(
    data,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    title: str,
    hue=None,
    xtick_rot: int = 65,
    max_len_xtick_labels: int = 25,
    xticklabels: dict = xticklabels_d,
    my_figsize: (int, int) = (10, 7),
):
    """
    Function to automate seaborn
    barplot plotting.
    """
    # Figure Size
    fig = plt.figure(figsize=my_figsize)

    # Bar Plot
    ax = sns.barplot(x=data[x], y=data[y], hue=hue)
    f = lambda x: textwrap.fill(x.get_text(), max_len_xtick_labels)
    ax.set_xticklabels(map(f, ax.get_xticklabels()), rotation=xtick_rot, **xticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.plot()
    
    
def plot_sns_countplot(
    data,
    x: str,
    plot_order,
    x_label: str,
    y_label: str,
    title: str,
    hue=None,
    xtick_rot: int = 65,
    max_len_xtick_labels: int = 25,
    xticklabels: dict = xticklabels_d,
    my_figsize: (int, int) = (10, 7),
):
    """
    Function to automate seaborn
    countplot plotting.
    """
    plt.figure(figsize=my_figsize)
    ax = sns.countplot(data=data, x=x, order=plot_order, hue=hue)
    f = lambda x: textwrap.fill(x.get_text(), max_len_xtick_labels)
    ax.set_xticklabels(map(f, ax.get_xticklabels()), rotation=xtick_rot, **xticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.plot()
    return ax


def plot_sns_jointplot(
    data, x: str, y: str, title: str, xlim=(-20, 850), ylim=(3, 5.1), my_figsize=(8, 5)
):
    """
    Function to automate seaborn
    jointplot plotting.
    """
    g = sns.JointGrid(data, x=x, y=y)
    g.plot_joint(sns.scatterplot, s=100, alpha=0.5)
    g.ax_marg_x.set_xlim(*xlim)
    g.ax_marg_y.set_ylim(*ylim)
    g.plot_marginals(sns.histplot, kde=True)
    g.fig.set_size_inches(my_figsize)
    g.fig.suptitle(title)

    g.fig.show()
    

def visualize_violinplot(df, x: str, y: str, hue: str = None):
    """
    Function to plot a violin plot with seaborn.
    """
    # Create the violin plot
    sns.violinplot(x=x, y=y, data=df, hue=hue)

    # Set the plot title and axes labels
    plt.title(f"{y.capitalize()} distribution by {x.capitalize()}")
    plt.xlabel(x.capitalize())
    plt.ylabel(y.capitalize())

    if hue is not None:
        plt.legend(title=hue, loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.title(
            f"{y.capitalize()} distribution by {x.capitalize()} and {hue.capitalize()}"
        )
    # Show the plot
    plt.show()
    

def plot_wordcloud(data, my_col, top_words: int = 50):
    """
    Function to use CountVectorizer to
    visualize top words in strings
    as a word cloud.
    """

    # get unique course titles
    series_variable = pd.Series(data[my_col].unique())

    # convert titles into lower case
    series_variable = series_variable.str.lower()

    # get word frequencies excluding the most common structures in English
    vectorizer = CountVectorizer(stop_words="english")
    word_count = vectorizer.fit_transform(series_variable)
    words_list = vectorizer.get_feature_names_out()
    counts_list = word_count.toarray().sum(axis=0)

    # get a list of tuples with first n most used words
    most_used_words = sorted(
        zip(words_list, counts_list), key=lambda x: x[1], reverse=True
    )[:top_words]

    # Generate a word cloud image
    text = " ".join([i[0] for i in most_used_words])
    wordcloud = WordCloud(stopwords=None, background_color="white").generate(text)

    # Display the generated image
    # with top-n words
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return most_used_words

    
def plot_count_percent_barplots_by_category(
    my_df, cat_col: str, my_col: str, my_title: str, my_order=None
):
    """
    Function to visualize two plots side by side.
    The first plot shows the total count for each category.
    The second plot shows the shares for each category.
    """

    fig = plt.figure(figsize=(16, 8))
    grid = GridSpec(1, 2)

    ax1 = fig.add_subplot(grid[0, 0])

    # Set the color palette for the countplot
    palette = {0: "darkgrey", 1: "steelblue"}

    sns.countplot(
        data=my_df.dropna(),
        x=cat_col,
        order=my_order,
        hue=my_col,
        palette=palette,
        ax=ax1,
        width=0.8,
    )
    ax1.set(xlabel=cat_col.capitalize(), ylabel="Count")

    ax2 = fig.add_subplot(grid[0, 1])

    # Calculate the share of positive ratings for each category
    share_positive = my_df.groupby(cat_col)[my_col].mean()

    if my_order is not None:
        share_positive = share_positive.loc[my_order].sort_values(ascending=False)

    # Calculate the share of negative ratings for each category
    share_negative = 1 - share_positive

    # Plot the stacked bars
    ax2.bar(
        share_positive.index,
        share_positive,
        color="steelblue",
        label="Positive",
    )
    ax2.bar(
        share_negative.index,
        share_negative,
        bottom=share_positive,
        color="darkgrey",
        label="Negative",
    )

    ylabel_name = my_col.replace("_", " ")

    ax2.set(xlabel=cat_col.capitalize(), ylabel=f"Share of {ylabel_name.capitalize()}")
    ax2.set_ylim(0, 1)
    ax2.legend()

    # Rotate the x-axis labels by 45 degrees
    ax1.tick_params(axis="x", labelrotation=45)
    ax2.tick_params(axis="x", labelrotation=45)

    fig.suptitle(my_title, fontsize=16)

    plt.subplots_adjust(wspace=0.3)

    plt.show()
