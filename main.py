import logging
import os
import uuid
import datetime
from fastapi import FastAPI
import inngest
from inngest.experimental import ai
from inngest.experimental.mocked import trigger
import inngest.fast_api
from dotenv import load_dotenv

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="chat_pdf",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    return {"hello": "world"}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])