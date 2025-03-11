"""
ChatGPT Dialogue Module
=======================

This module provides a dialogue system based on the GPT-3 model from OpenAI. It uses the
OpenAI API to generate responses to user input. The module is incremental, so that it
produces hypotheses for each user input. The hypotheses are being "committed" once the
input is final.
"""

import retico_core
import threading
import openai
from openai import OpenAI
import time
import os


class ChatGPTDialogueModule(retico_core.AbstractModule):
    """ChatGPT Dialogue Module that uses the OpenAI API to generate responses to user
    input. The ChatGPTDialogueModule is not running locally, but uses the OpenAI API to
    generate responses.
    """

    @staticmethod
    def name():
        return "ChatGPT Dialogue Module"

    @staticmethod
    def description():
        return "ChatGPT Dialogue Module that uses the OpenAI API to generate responses to user input."

    @staticmethod
    def input_ius():
        return [retico_core.text.TextIU]

    @staticmethod
    def output_iu():
        return retico_core.text.TextIU

    def __init__(
        self,
        system_prompt,
        model="gpt-4o-mini",
        max_tokens=1000,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        api_key=None,
        base_url=None,
        max_history=9,
        **kwargs,
    ):
        """Initializes the ChatGPT Dialogue Module with the given system prompt and gpt
        parameters. The API key can be passed as an argument or set as an environment
        variable. For a more specific description of the GPT parameters, see the OpenAI
        documentation: https://platform.openai.com/docs/api-reference/chat

        Args:
            system_prompt (str): The system prompt that is used to prime the GPT model.
            model (str, optional): The OpenAI model that should be used. Defaults to "gpt-3.5-turbo".
            max_tokens (int, optional): The maximum number of tokens. Defaults to 1000.
            temperature (float, optional): The temperature of the GPT model. Defaults to 0.7.
            top_p (int, optional): Nucleus sampling parameter. Defaults to 1.
            frequency_penalty (int, optional): Penalize tokens based on frequency. Defaults to 0.
            presence_penalty (int, optional): Penalize tokens based on whether they appear in the text. Defaults to 0.
            api_key (str, optional): OpenAI API key. Defaults to None.
            max_history (int, optional): The maximum number of previous user inputs that are used to prime the GPT model. Defaults to 9.
        """
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_history = max_history
        self.history = []

        self.run_loop = False
        self.current_text = ""

        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if base_url is not None:
            self.base_url = base_url
        elif os.getenv("OPENAI_BASE_URL") is not None:
            self.base_url = os.getenv("OPENAI_BASE_URL")
        else:
            self.base_url = "https://api.openai.com/v1"


        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
            )

    def setup(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
            )
        

    def prepare_run(self):
        self.run_loop = True
        threading.Thread(target=self._generate_loop).start()

    def shutdown(self):
        self.run_loop = False

    def input_text(self):
        """Joins the current input into a single string and returns it.

        Returns:
            str: The current input as a single string.
        """
        return " ".join([iu.text for iu in self.current_input])

    def output_text(self):
        """Joins the current output into a single string and returns it.

        Returns:
            str: The current output as a single string.
        """
        return " ".join([iu.text for iu in self.current_output])

    def process_update(self, update_message):
        """Processes new incoming messgaes and activates the GPT model to generate a
        response. The response is then added to the history and the response is passed
        to the next module.

        Args:
            update_message (retico_core.UpdateMessage): The incoming UpdateMessage.
        """
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.current_input.append(iu)
                self.latest_input_iu = iu
            elif ut == retico_core.UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == retico_core.UpdateType.COMMIT:
                self.commit(iu)

            if self.current_input and self.input_committed():
                if len(self.input_text().strip()) < 3:
                    self.current_input = []
                    return
                current_text = self.input_text()
                self.current_input = []
                self.current_text = current_text

    def get_response(self, input):
        """Generates a response to the given input. The response is generated by the
        OpenAI API.

        If the context length is exceeded, the max_history is reduced by one and the
        function is called again.

        Args:
            input (str): The input for which a response should be generated.
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        for previous_input, previous_response in self.history[-self.max_history :]:
            messages.append({"role": "user", "content": previous_input})
            messages.append({"role": "assistant", "content": previous_response})
        messages.append({"role": "user", "content": input})
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stream=True,
            )
            return completion
        except openai.BadRequestError as e:
            if e.code == "context_length_exceeded" or "maximum context length" in str(e.message):
                self.max_history -= 1
                if self.max_history == 0 or len(messages) <= 2:
                    print("Context length exceeded. Last response too long.")
                    return []
                else:
                    print(
                        f"Context length exceeded. Reducing max_history to {self.max_history - 1}"
                    )
                    return self.get_response(input)
            else:
                print("Error: ", e)
                return []
        except openai.NotFoundError as e:
            #list avalible model
            models = self.client.models.list().data
            models = [model["id"] if type(model) == dict else model.id for model in models]

            print(f"No model with the id {self.model} found. Available models are: {models}")
            return []

            

        except Exception as e:
            print("Error: ", e)

    def _generate_loop(self):
        """The GPT model is called in a loop to generate responses. The loop is
        terminated once the run_loop flag is set to False.
        """
        while self.run_loop:
            if self.current_text != "":
                completion = self.get_response(self.current_text)
                current_content = ""
                for chunk in completion:
                    choice = chunk.choices[0]
                    if type(choice) != dict:
                        choice = choice.to_dict()#to make it more compatible across servers
                    if choice["delta"].get("content"):
                        current_content += choice["delta"]["content"]
                        if " " in current_content:
                            split = current_content.split(" ")
                            for i, word in enumerate(split):
                                if i == len(split) - 1:
                                    current_content = word
                                    break
                                if word != "":
                                    current_iu = self.create_iu(self.latest_input_iu)
                                    current_iu.text = word
                                    self.current_output.append(current_iu)
                                    um = retico_core.UpdateMessage.from_iu(
                                        current_iu, retico_core.UpdateType.ADD
                                    )
                                    self.append(um)

                    if choice["finish_reason"] == "stop":
                        um = retico_core.UpdateMessage()
                        if current_content != "":
                            current_iu = self.create_iu(self.latest_input_iu)
                            current_iu.text = current_content
                            self.current_output.append(current_iu)
                            um.add_iu(current_iu, retico_core.UpdateType.ADD)
                        for iu in self.current_output:
                            self.commit(iu)
                            um.add_iu(iu, retico_core.UpdateType.COMMIT)
                        self.append(um)
                        self.history.append((self.current_text, self.output_text()))
                        self.current_output = []
                        self.current_text = ""
            time.sleep(0.1)
