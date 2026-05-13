from kedro.pipeline import Pipeline, node

from .nodes import (
    split_data,
    train_model,
    predict,
    compute_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["features", "params:training"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),

            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:training"],
                outputs="model",
                name="train_model_node",
            ),

            node(
                func=predict,
                inputs=["model", "X_test"],
                outputs="predictions",
                name="predict_node",
            ),

            node(
                func=compute_metrics,
                inputs=["y_test", "predictions"],
                outputs="metrics",
                name="metrics_node",
            ),
        ]
    )