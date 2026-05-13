from kedro.pipeline import Pipeline

from currency_movement_prediction_pipeline.pipelines.feature_eng.pipeline import (
    create_pipeline as feature_pipeline,
)

from currency_movement_prediction_pipeline.pipelines.training.pipeline import (
    create_pipeline as training_pipeline,
)

from currency_movement_prediction_pipeline.pipelines.inference.pipeline import (
    create_pipeline as inference_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:

    feature_eng = feature_pipeline()

    training = training_pipeline()

    inference = inference_pipeline()

    return {

        "__default__": (
            feature_eng + training
        ),

        "training": (
            feature_eng + training
        ),

        "inference": (
            feature_eng + inference
        ),
    }