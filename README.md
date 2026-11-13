# retico-chatgpt
A ReTiCo module that uses an LLM through the OpenAI API to generate responses to user input incrementally. It works with OpenAI models and with any OpenAI-compatible server, such as Groq, Mistral or a self-hosted vLLM server.

## Installation and Requirements

To use the ChatGPT module you first need to install the retico-chatgpt package:
* Install retico-chatgpt:
```pip install retico-chatgpt```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-chatgpt.git```

This also installs its Python dependencies (`retico-core` and `openai`).

### API key and server
The module needs an API key for the server it talks to. Either pass it as the `api_key` argument, or set the `OPENAI_API_KEY` environment variable.

By default the module talks to OpenAI (`https://api.openai.com/v1`). To use another OpenAI-compatible server, pass `base_url` or set the `OPENAI_BASE_URL` environment variable:

```bash
OPENAI_API_KEY=<your-api-key>
OPENAI_BASE_URL="https://api.openai.com/v1"      # OpenAI (default)
OPENAI_BASE_URL="https://api.groq.com/openai/v1" # Groq
OPENAI_BASE_URL="http://0.0.0.0:8000/v1"         # vLLM (its default API key is "EMPTY")
```

Or set them from Python before creating the module:
```python
import os
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
```

## Modules

### `ChatGPTDialogueModule`
Subscribes to `TextIU` and outputs `TextIU`. Once the user's input is committed, it sends the input to the LLM and streams the response forward word by word, committing the words once the response is finished. Previous turns are kept as conversation history.

Inputs shorter than 3 characters are ignored.

#### Arguments:
* `system_prompt` (str): the system prompt used to prime the model; required
* `model` (str): the model to use; defaults to `"gpt-4o-mini"`
* `max_tokens` (int): the maximum number of tokens in a response; defaults to `1000`
* `temperature` (float): the sampling temperature; defaults to `0.7`
* `top_p` (float): nucleus sampling parameter; defaults to `1`
* `frequency_penalty` (float): penalize tokens based on how often they appear; defaults to `0`
* `presence_penalty` (float): penalize tokens based on whether they already appear; defaults to `0`
* `api_key` (str): the API key; defaults to `None` (uses `OPENAI_API_KEY`)
* `base_url` (str): the URL of the OpenAI-compatible server; defaults to `None` (uses `OPENAI_BASE_URL`, or OpenAI if that isn't set)
* `max_history` (int): the maximum number of previous turns sent to the model as context; defaults to `9`. If the model's context length is exceeded, this is reduced automatically

For more details on the model parameters, see the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat).

### Example
**Note:** The example uses `MicrophoneModule` and `TextPrinterModule` from `retico_core` and `Wav2VecASRModule` from [retico-wav2vecasr](https://github.com/retico-team/retico-wav2vecasr) (`pip install retico-wav2vecasr`). It loads the API key from a `.env` file with [python-dotenv](https://pypi.org/project/python-dotenv/) (`pip install python-dotenv`).

```python
import retico_core
import retico_wav2vecasr
import retico_chatgpt
import dotenv

dotenv.load_dotenv()

INSTRUCTIONS = """You are an AI assistant that is an expert in alcoholic beverages.
You know about cocktails, wines, spirits and beers.
You can provide advice on drink menus, cocktail ingredients, how to make cocktails, and anything else related to alcoholic drinks.
If you are unable to provide an answer to a question, please respond with the phrase "I'm just a simple barman, I can't help with that."
Please aim to be as helpful, creative, and friendly as possible in all of your responses.
Do not use any external URLs in your answers. Do not refer to any blogs in your answers.
You are interacting with spoken words. Be very brief and do not format your responses.
Do not use dashes in lists, and do not use any other formatting.
Write out numbers as words, for example use "one" instead of "1".
"""

mic = retico_core.audio.MicrophoneModule()
asr = retico_wav2vecasr.Wav2VecASRModule(language="en")
chatgpt = retico_chatgpt.ChatGPTDialogueModule(system_prompt=INSTRUCTIONS)
printer = retico_core.debug.TextPrinterModule()

mic.subscribe(asr)
asr.subscribe(chatgpt)
chatgpt.subscribe(printer)

retico_core.network.run(mic)

print("Network is running")
input()

retico_core.network.stop(mic)
```

The `.env` file goes in the folder you run the script from:
```bash
OPENAI_API_KEY=<your-api-key>
```

### Hosting a local model with vLLM
1. In the same or another environment, install vLLM: `pip install vllm`
2. Serve a model from Hugging Face:
```
vllm serve HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints --port 8000
```
3. Set `OPENAI_BASE_URL="http://0.0.0.0:8000/v1"` and `OPENAI_API_KEY="EMPTY"` (see [API key and server](#api-key-and-server)), and pass the served model's name as `model`.