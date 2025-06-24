from app.llm_client import ask_gpt
from unittest.mock import patch, MagicMock


def test_ask_gpt_returns_expected_response():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="Mocked answer!"))]

    with patch("app.llm_client.openai.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = fake_response

        result = ask_gpt('What time is it?')
        assert result == "Mocked answer!"


def test_ask_gpt_sends_prompt():
    with patch("app.llm_client.openai.OpenAI") as mock_openai:
        mock_create = mock_openai.return_value.chat.completions.create
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OK"))]
        )

        prompt = "Say hello"
        ask_gpt(prompt)

        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        assert kwargs["messages"][0]["content"] == prompt



