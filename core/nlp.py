from openai import OpenAI


class LanguageModel:
    def pseudo_to_text(self, pseudo_text: str) -> str:
        return pseudo_text


class ChatGptModel(LanguageModel):
    models = {
        "gpt-4-0125-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-1106-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-1106-vision-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4": {
            "input": 0.03,
            "output": 0.06
        },
        "gpt-4-0613": {
            "input": 0.03,
            "output": 0.06
        },
        "gpt-4-32k": {
            "input": 0.06,
            "output": 0.12
        },
        "gpt-3.5-turbo-0125": {
            "input": 0.0005,
            "output": 0.0015
        },
        "gpt-3.5-turbo-instruct": {
            "input": 0.0015,
            "output": 0.0020
        },
        "gpt-3.5-turbo-16k-0613": {
            "input": 0.0030,
            "output": 0.0040
        },
        "gpt-3.5-turbo-1106": {
            "input": 0.0010,
            "output": 0.0020
        },
        "gpt-3.5-turbo-0613": {
            "input": 0.0015,
            "output": 0.0020
        },
        "gpt-3.5-turbo-0301": {
            "input": 0.0015,
            "output": 0.0020
        }
    }

    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo-0613', temperature=0.5):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
        self._in_token_usage = 0
        self._out_token_usage = 0

    def pseudo_to_text(self, pseudo_text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content":
                        """
                        You are a data expert capable of converting pseudo English sentences into a meaningful paragraph
                        without losing any meaning or adding anything new. Group information based on the assigned groups.

                        Example:
                        x-[relation 1]->y
                        z-[relation 2]->x
                        m-[relation 3]->n

                        Output:
                        X has this relation 1 with Y, Z shares a relation 2 with X.
                        Moreover, m has relation 3 with n.
                        """
                },
                {"role": "user", "content": pseudo_text},
            ]
        )
        self._in_token_usage += response.usage.prompt_tokens
        self._out_token_usage += response.usage.completion_tokens
        return response.choices[0].message.content

    @property
    def cost(self) -> float:
        model_pricing = self.models.get(self.model)

        in_tokens = self._in_token_usage / 1000
        out_tokens = self._out_token_usage / 1000

        return in_tokens * model_pricing['input'] + out_tokens * model_pricing["output"]
