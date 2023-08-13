import pandas as pd
import numpy as np
import altair as alt
from sklearn.manifold import TSNE

from coati.math_tools.plots import image_formatter2, wrapped_get_smiles_image


def embed_altair(
    df,
    tooltip_fields=["smiles", "mol_index"],
    selector_field="library",
    quantity="FOLDOVER_ALL_MEDIAN",
    image_tooltip=True,
    emb_field="emb",
    smiles_field="smiles",
    width=1024,
    height=768,
):
    """
    Args:
        df: a dataframe with an 'emb' column. (numpy)
        quantity: a quantity which will be plotted.
    """
    df["image"] = (
        df[smiles_field].apply(wrapped_get_smiles_image).apply(image_formatter2).copy()
    )
    df["mol_index"] = range(df.shape[0])

    if not selector_field is None:
        substrate_frame = pd.DataFrame(
            {selector_field: df[selector_field].unique().tolist()}
        )
        selection = alt.selection_multi(fields=[selector_field])
        color1 = alt.condition(
            selection, alt.Color(selector_field + ":N"), alt.value("lightgray")
        )
        substrate_selector = (
            alt.Chart(substrate_frame)
            .mark_rect()
            .encode(y=selector_field, color=color1)
            .add_selection(selection)
        )

    tooltip = [alt.Tooltip(field=f, title=f) for f in tooltip_fields]
    if image_tooltip:
        tooltip.append("image")

    embs = np.stack(df[emb_field].values.tolist(), 0)
    X_embedded = TSNE(n_components=2, learning_rate=100, init="random").fit_transform(
        embs
    )
    df.loc[:, "X"] = X_embedded[:, 0]
    df.loc[:, "Y"] = X_embedded[:, 1]

    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                "X:Q",
                scale=alt.Scale(zero=False, domain=[df.X.min() - 1, df.X.max() + 1]),
            ),
            y=alt.X(
                "Y:Q",
                scale=alt.Scale(zero=False, domain=[df.Y.min() - 1, df.Y.max() + 1]),
            ),
            color=alt.Color(
                quantity + ":Q",
                scale=alt.Scale(range=["orange", "blue"]),  # type='log',
            ),
            opacity=alt.value(0.5),
            tooltip=tooltip,
        )
        .properties(title="", width=int(width), height=height)
    )

    if not selector_field is None:
        return alt.hconcat(substrate_selector, chart.transform_filter(selection))
    else:
        return chart
