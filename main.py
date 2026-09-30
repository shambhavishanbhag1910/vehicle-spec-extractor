import json
from dotenv import load_dotenv

from src.pipeline import (
    VehicleSpecificationPipeline
)


load_dotenv()


PDF_PATH = (
    "data/sample-service-manual.pdf"
)


pipeline = VehicleSpecificationPipeline()

pipeline.ingest(PDF_PATH)


while True:

    query = input(
        "\nEnter specification query "
        "(or 'exit'): "
    )

    if query.lower() == "exit":
        break

    result = pipeline.ask(query)

    print(
        json.dumps(
            result,
            indent=2
        )
    )