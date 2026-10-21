import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

class SageMakerPipelineBuilder:
    def __init__(self, pipeline_name: str, model_group_name: str):
        self.pipeline_session = PipelineSession()
        
        # Lê a Role de forma dinâmica por variável de ambiente (Sem valores fixos)
        self.role = os.getenv("SAGEMAKER_ROLE")
        if not self.role:
            raise ValueError(
                "A variável de ambiente 'SAGEMAKER_ROLE' não está definida. "
                "Defina-a no seu terminal antes de executar: export SAGEMAKER_ROLE='arn:aws:iam::...'"
            )

        self.bucket = self.pipeline_session.default_bucket()
        self.pipeline_name = pipeline_name
        self.model_group_name = model_group_name

        self.sklearn_image = sagemaker.image_uris.retrieve("sklearn", self.pipeline_session.boto_region_name, version="1.2-1")
        self.xgboost_image = sagemaker.image_uris.retrieve("xgboost", self.pipeline_session.boto_region_name, version="1.7-1")

    def build_processing_step(self) -> ProcessingStep:
        processor = ScriptProcessor(
            image_uri=self.sklearn_image,
            role=self.role,
            instance_type="ml.m5.large",
            instance_count=1,
            command=["python3"],
            sagemaker_session=self.pipeline_session
        )
        return ProcessingStep(
            name="TitanicTransformStep",
            processor=processor,
            code="preprocessing.py",
            inputs=[
                ProcessingInput(
                    source=f"s3://{self.bucket}/titanic/titanic.csv", 
                    destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
                ProcessingOutput(output_name="transformer", source="/opt/ml/processing/output/transformer")
            ]
        )

    def build_training_step(self, transform_step: ProcessingStep) -> TrainingStep:
        estimator = Estimator(
            image_uri=self.xgboost_image,
            role=self.role,
            instance_type="ml.m5.large",
            instance_count=1,
            entry_point="train.py",
            output_path=f"s3://{self.bucket}/models",
            hyperparameters={"seed": 42, "max-depth": 3, "n-estimators": 100},
            sagemaker_session=self.pipeline_session
        )
        return TrainingStep(
            name="TitanicTrainStep",
            estimator=estimator,
            inputs={
                "train": transform_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            }
        )

    def build_register_step(self, train_step: TrainingStep) -> ModelStep:
        model = Model(
            image_uri=self.xgboost_image,
            model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            role=self.role,
            sagemaker_session=self.pipeline_session
        )
        return ModelStep(
            name="TitanicModelVersioningStep",
            step_args=model.register(
                content_types=["text/csv"],
                response_types=["application/json"],
                model_package_group_name=self.model_group_name,
                approval_status="PendingManualApproval"
            )
        )

    def create_and_run(self):
        transform_step = self.build_processing_step()
        train_step = self.build_training_step(transform_step)
        register_step = self.build_register_step(train_step)

        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[transform_step, train_step, register_step],
            sagemaker_session=self.pipeline_session
        )
        
        pipeline.upsert(role_arn=self.role)
        pipeline.start()
        print(f"Pipeline {self.pipeline_name} iniciado com sucesso.")

if __name__ == "__main__":
    builder = SageMakerPipelineBuilder(
        pipeline_name="TitanicPipelineOOP",
        model_group_name="titanic-model-group"
    )
    builder.create_and_run()
