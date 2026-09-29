import streamlit as st

from dotenv import load_dotenv

from src.pipeline import (
    VehicleSpecificationPipeline
)


load_dotenv()


@st.cache_resource
def load_pipeline():

    pipeline = (
        VehicleSpecificationPipeline()
    )

    pipeline.ingest(
        "data/sample-service-manual.pdf"
    )

    return pipeline


st.title(
    "2014 F-150 Specification Extractor"
)

st.write(
    "Ask technical specification questions "
    "from the service manual."
)


pipeline = load_pipeline()


query = st.text_input(
    "Specification question"
)


if st.button("Extract"):

    if query:

        with st.spinner(
            "Searching service manual..."
        ):

            result = pipeline.ask(query)

        st.json(result)