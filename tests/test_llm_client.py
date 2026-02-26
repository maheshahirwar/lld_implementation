from types import SimpleNamespace

from app.llm_client import LLMConfigurationError, OllamaLLMClient, build_llm_client


def test_build_llm_client_ollama() -> None:
    settings = SimpleNamespace(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
    )

    client = build_llm_client(settings)
    assert isinstance(client, OllamaLLMClient)


def test_build_llm_client_openai_requires_key() -> None:
    settings = SimpleNamespace(
        llm_provider="openai",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
    )

    try:
        build_llm_client(settings)
    except LLMConfigurationError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected LLMConfigurationError")
