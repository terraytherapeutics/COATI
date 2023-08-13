# altairified versions of plotting routines.
import altair as alt
import pandas as pd
from sklearn.metrics import roc_curve, auc


alt.data_transformers.disable_max_rows()


def roc_plot(source, y_true="y", y_pred="y_pred", partition_col=None, chart_args=None):

    if chart_args is None:
        chart_args = {"height": 400, "width": 400}

    if not (partition_col is None):
        train_fpr, train_tpr, _ = roc_curve(
            y_true=source[source[partition_col] == "train"][y_true],
            y_score=source[source[partition_col] == "train"][y_pred],
        )
        train_auc = round(auc(train_fpr, train_tpr), 3)
        train_df = pd.DataFrame(
            {"False Positive Rate": train_fpr, "True Positive Rate": train_tpr}
        )
        train_df[partition_col] = f"train - auROC: {train_auc}"

        test_fpr, test_tpr, _ = roc_curve(
            y_true=source[source[partition_col] == "test"][y_true],
            y_score=source[source[partition_col] == "test"][y_pred],
        )
        test_auc = round(auc(test_fpr, test_tpr), 3)
        test_df = pd.DataFrame(
            {"False Positive Rate": test_fpr, "True Positive Rate": test_tpr}
        )
        test_df[partition_col] = f"test - auROC: {test_auc}"

        return (
            alt.Chart(pd.concat([train_df, test_df]), **chart_args)
            .mark_line()
            .encode(
                x="False Positive Rate", y="True Positive Rate", color=partition_col
            )
        )
    else:
        test_fpr, test_tpr, _ = roc_curve(
            y_true=source[y_true],
            y_score=source[y_pred],
        )
        test_auc = round(auc(test_fpr, test_tpr), 3)
        test_df = pd.DataFrame(
            {"False Positive Rate": test_fpr, "True Positive Rate": test_tpr}
        )
        return (
            alt.Chart(test_df, **chart_args)
            .mark_line()
            .encode(x="False Positive Rate", y="True Positive Rate")
            .properties(title=f"auROC: {test_auc}")
        )

