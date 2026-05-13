from kedro.pipeline import Pipeline, node

from .nodes import (
    load_data,
    clean_data,
    filter_data,
    rename_columns,
    create_target,
    get_features,
)


def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline(

        [

            # =========================
            # LOAD DATA
            # =========================
            node(
                func=load_data,
                inputs="train_data",
                outputs="raw_data",
                name="load_data_node",
            ),

            # =========================
            # CLEAN DATA
            # =========================
            node(
                func=clean_data,
                inputs="raw_data",
                outputs="cleaned_data",
                name="clean_data_node",
            ),

            # =========================
            # FILTER DATA
            # =========================
            node(
                func=filter_data,
                inputs="cleaned_data",
                outputs="filtered_data",
                name="filter_data_node",
            ),

            # =========================
            # RENAME COLUMNS
            # =========================
            node(
                func=rename_columns,
                inputs=[
                    "filtered_data",
                    "params:feature_engineering.rename_columns",
                ],
                outputs="renamed_data",
                name="rename_columns_node",
            ),

            # =========================
            # CREATE TARGET
            # IMPORTANT:
            # AFTER RENAME
            # =========================
            node(
                func=create_target,
                inputs="renamed_data",
                outputs="target_data",
                name="create_target_node",
            ),

            # =========================
            # FEATURE ENGINEERING
            # =========================
            node(
                func=get_features,
                inputs=[
                    "target_data",
                    "params:feature_engineering.lag_params",
                ],
                outputs=[
                    "features",
                    "timestamps",
                ],
                name="feature_engineering_node",
            ),

        ]

    )