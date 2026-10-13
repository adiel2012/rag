"""The two model providers: a local Ollama server, or Amazon Bedrock."""
import logging
import time
from typing import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from .config import Settings

log = logging.getLogger(__name__)


def build_llm(s: Settings) -> Runnable:
    if s.provider == "ollama":
        from langchain_ollama import ChatOllama
        # num_ctx is set explicitly: Ollama's default window is small and it silently
        # truncates prompts that do not fit, which would cut off instructions or context.
        return ChatOllama(
            model=s.llm_model, base_url=s.ollama_base_url, temperature=s.temperature,
            num_ctx=s.num_ctx, reasoning=s.thinking,
        )
    from langchain_aws import ChatBedrockConverse
    kwargs = {} if s.temperature is None else {"temperature": s.temperature}
    return ChatBedrockConverse(model=s.llm_model, region_name=s.aws_region, **kwargs)


def structured(llm, schema, s: Settings) -> Runnable:
    """The model constrained to return `schema`."""
    if s.provider == "ollama":
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)   # Bedrock: tool calling


def build_embeddings(s: Settings) -> Embeddings:
    if s.provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_base_url)
    from langchain_aws import BedrockEmbeddings
    return BedrockEmbeddings(model_id=s.embed_model, region_name=s.aws_region)


def ensure_models(s: Settings, report: Callable[[str, str], None], wait_seconds: int = 120) -> None:
    """Ollama: wait for the server and download the models that are missing.
    Bedrock: nothing to prepare; a wrong model ID or missing credentials fail on first use."""
    if s.provider != "ollama":
        return
    import ollama
    client = ollama.Client(host=s.ollama_base_url)

    for _ in range(wait_seconds):
        try:
            client.list()
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError(f"No Ollama server answered at {s.ollama_base_url}.")

    available = {m.model for m in client.list().models}
    for model in dict.fromkeys([s.llm_model, s.embed_model]):
        if model in available or f"{model}:latest" in available:
            continue
        log.info("Downloading %s ...", model)
        last = None
        for part in client.pull(model, stream=True):
            if part.total and part.total > 100_000_000:     # report only the large files
                step = (part.digest, 10 * int(10 * (part.completed or 0) / part.total))
                if step != last:
                    last = step
                    report("pulling model", f"{model}: {step[1]}%")
                    log.info("  %s: %s%%", model, step[1])
