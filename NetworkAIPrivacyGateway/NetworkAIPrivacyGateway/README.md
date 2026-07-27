# Network AI Privacy Gateway

A Streamlit-based demonstration of a privacy gateway that tokenizes sensitive network diagnostics before sending them to a Large Language Model (LLM). The application protects enterprise network identifiers, masks them, sends only masked data to the LLM, and then restores original values in the final report.

## Features

- Streamlit dashboard for network troubleshooting
- Ping, traceroute, and TCP port connectivity checks
- Sensitive data masking engine with reversible tokenization
- Support for OpenAI and Ollama providers
- Prompt builder for secure LLM requests
- Session-state driven UI with download support
- Configurable pipeline using environment variables

## Project Structure

```
NetworkAIPrivacyGateway/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── core/
│   ├── network_tools.py
│   ├── masker.py
│   ├── llm_client.py
│   ├── prompt_builder.py
│   ├── logger.py
│   ├── utils.py
│   └── constants.py
│
├── assets/
├── logs/
└── tests/
```

## Requirements

- Python 3.10+
- Streamlit
- OpenAI Python client
- Requests
- Pydantic

## Installation

```powershell
cd "d:\Python Projects\NetworkAIPrivacyGateway"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

```powershell
streamlit run app.py
```

## Environment

Create a `.env` file with values if you want to preconfigure settings:

```ini
LLM_PROVIDER=openai
LLM_SERVER_URL=https://api.openai.com
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=your-api-key
```

## Testing

Run unit tests with pytest:

```powershell
pytest
```

## Security Notes

- API keys are never logged
- Shell injection is avoided by using subprocess command lists
- IP validation is enforced before diagnostics
- Sensitive values are masked before being sent to LLM
