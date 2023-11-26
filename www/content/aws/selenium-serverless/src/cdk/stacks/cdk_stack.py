from aws_cdk import aws_lambda as lambda_
from aws_cdk import Stack
import aws_cdk as cdk
from constructs import Construct


class CdkStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        selenium_function = lambda_.DockerImageFunction(
            self,
            "SeleniumFunction",
            code=lambda_.DockerImageCode.from_image_asset("../app", file="Dockerfile"),
            function_name=f"test-SeleniumFunction",
            timeout=cdk.Duration.minutes(1),
            memory_size=2048,
        )
