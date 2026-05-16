"""Tests that switch_model does not inherit stale context_length overrides."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


def _make_agent_with_compressor(config_context_length=None) -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    # Primary model settings
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "dummy"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    # Store the initial config_context_length override used at agent construction.
    agent._config_context_length = config_context_length

    # Context compressor with primary model values
    compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="dummy",
        provider="openrouter",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor

    # For switch_model
    agent._primary_runtime = {}

    return agent


def test_switch_model_uses_current_config_context_length():
    """When switching models, the live global model.context_length override is passed through."""
    agent = _make_agent_with_compressor(config_context_length=None)

    with (
        patch("agent.model_metadata.get_model_context_length", return_value=131_072) as mock_ctx_len,
        patch("hermes_cli.config.load_config", return_value={"model": {"context_length": 32_768}}),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
    ):
        agent.switch_model("new-model", "openrouter", api_key="dummy", base_url="https://openrouter.ai/api/v1")

    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 32_768
    assert agent._config_context_length == 32_768
    assert agent.context_compressor.model == "new-model"
    assert agent.context_compressor.context_length == 131_072


def test_switch_model_does_not_reuse_stale_context_length():
    """A previous model's cached context override must not mask the new model's context."""
    agent = _make_agent_with_compressor(config_context_length=32_768)

    with (
        patch("agent.model_metadata.get_model_context_length", return_value=131_072) as mock_ctx_len,
        patch("hermes_cli.config.load_config", return_value={"model": {}}),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
    ):
        agent.switch_model("new-model", "openrouter", api_key="dummy", base_url="https://openrouter.ai/api/v1")

    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") is None
    assert agent._config_context_length is None
    assert agent.context_compressor.model == "new-model"
    assert agent.context_compressor.context_length == 131_072


def test_switch_model_without_config_context_length():
    """When switching models without config override, config_context_length should be None."""
    agent = _make_agent_with_compressor(config_context_length=None)

    with (
        patch("agent.model_metadata.get_model_context_length", return_value=128_000) as mock_ctx_len,
        patch("hermes_cli.config.load_config", return_value={"model": {}}),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
    ):
        # Switch model
        agent.switch_model("new-model", "openrouter", api_key="dummy", base_url="https://openrouter.ai/api/v1")

        # Verify get_model_context_length was called with None
        mock_ctx_len.assert_called_once()
        call_kwargs = mock_ctx_len.call_args.kwargs
        assert call_kwargs.get("config_context_length") is None
