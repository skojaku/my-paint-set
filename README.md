# Before you go 
The following snipes require
- seaborn
- matplotlib.pyplot 

## Joint distribution 
```python
def hexbin(x, y, **kwargs):
    rs, _ = stats.pearsonr(x, y)
    # rs = metrics.r2_score(x, y)

    xmin = np.quantile(x, 0.0)
    xmax = np.quantile(x, 1)

    plt.hexbin(
        x,
        y,
        gridsize=50,
        edgecolors="none",
        cmap="Greys",
        linewidths=0.1,
        mincnt=5,
        **kwargs
    )
    ax = plt.gca()
    ax.plot(
        [xmin, xmax], [xmin, xmax], color=sns.color_palette().as_hex()[3], lw=3, ls=":"
    )

    xmin = np.quantile(x, 0.01)
    xmax = np.quantile(x, 1 - 0.001)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.annotate(
        r"$\rho=%.2f$" % rs,
        (0.05, 1 - 0.05),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=20,
    )

    return ax
```

## Stacked barchar
```python 
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(10, 5))

df_year.pivot(index="year", columns="journal_code", values="sz").plot.bar(
    stacked=True, cmap="tab20", width=1, ax=ax
)

xticklabels = np.array([int(float(l.get_text())) for l in ax.get_xticklabels()])
xtick_ids = np.where(xticklabels % 10 == 0)[0]
xtick_labels = xticklabels[xtick_ids]

ax.set_xticks(xtick_ids)
ax.set_xticklabels(xtick_labels, rotation=0)
ax.legend(frameon=False, ncol=2, fontsize=12)
sns.despine()
```

### How to fix legend cut off when saving a figure
```python
fig.savefig(output_file, dpi=300, bbox_inches="tight")
```

### Text wrapping

```python
import textwrap

f = lambda x: textwrap.fill(x.get_text(), 40)
ax.set_yticklabels(map(f, ax.get_yticklabels()), fontsize=11)
```

# 3D interactive plot in VSCODE


```python
%matplotlib ipympl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# creating figure
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# creating the plot
cmap = sns.color_palette().as_hex()
plot_geeks = ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    color=[cmap[i] for i in np.unique(labels, return_inverse=True)[1]],
)
plt.show()
```

# Plotting multiple distributions by categories

```python 
from textwrap import wrap

def categorical_kde_plot(
    df,
    variable,
    category,
    category_order=None,
    horizontal=False,
    rug=True,
    figsize=None,
    cmap=None,
    max_label_width=None,
):
    """Draw a categorical KDE plot

    Parameters
    ----------
    df: pd.DataFrame
        The data to plot
    variable: str
        The column in the `df` to plot (continuous variable)
    category: str
        The column in the `df` to use for grouping (categorical variable)
    horizontal: bool
        If True, draw density plots horizontally. Otherwise, draw them
        vertically.
    rug: bool
        If True, add also a sns.rugplot.
    figsize: tuple or None
        If None, use default figsize of (7, 1*len(categories))
        If tuple, use that figsize. Given to plt.subplots as an argument.
    """
    if category_order is None:
        categories = list(df[category].unique())
    else:
        categories = category_order[:]

    figsize = (7, 1.0 * len(categories))

    fig, axes = plt.subplots(
        nrows=len(categories) if horizontal else 1,
        ncols=1 if horizontal else len(categories),
        figsize=figsize[::-1] if not horizontal else figsize,
        sharex=horizontal,
        sharey=not horizontal,
    )

    for i, (cat, ax) in enumerate(zip(categories, axes)):
        sns.kdeplot(
            data=df[df[category] == cat],
            x=variable if horizontal else None,
            y=None if horizontal else variable,
            # kde kwargs
            bw_adjust=0.5,
            clip_on=False,
            fill=True,
            alpha=1,
            linewidth=1.5,
            ax=ax,
            color="lightslategray" if cmap is None else cmap[cat],
        )

        keep_variable_axis = (i == len(fig.axes) - 1) if horizontal else (i == 0)

        if rug:
            sns.rugplot(
                data=df[df[category] == cat],
                x=variable if horizontal else None,
                y=None if horizontal else variable,
                ax=ax,
                color="black",
                height=0.025 if keep_variable_axis else 0.04,
            )

        _format_axis(
            ax,
            cat,
            horizontal,
            keep_variable_axis=keep_variable_axis,
            max_label_width=max_label_width,
        )

    plt.tight_layout()
    return fig, axes


def _format_axis(
    ax, category, horizontal=False, keep_variable_axis=True, max_label_width=None
):

    # Remove the axis lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if max_label_width is not None:
        category = "\n".join(wrap(category, max_label_width))

    if horizontal:
        ax.set_ylabel(None)
        lim = ax.get_ylim()
        ax.set_yticks([(lim[0] + lim[1]) / 2])
        ax.set_yticklabels([category])
        if not keep_variable_axis:
            ax.get_xaxis().set_visible(False)
            ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel(None)
        lim = ax.get_xlim()
        ax.set_xticks([(lim[0] + lim[1]) / 2])
        ax.set_xticklabels([category])
        if not keep_variable_axis:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)


cmap = sns.color_palette().as_hex()
cmap = {h: cmap[i] for i, h in enumerate(hue_order)}
cmap["All"] = "lightslategray"

fig, axes = categorical_kde_plot(
    plot_data,
    variable="distance",
    category="category",
    category_order=["All"] + hue_order,
    horizontal=True,
    cmap=cmap,
    max_label_width=30,
```
