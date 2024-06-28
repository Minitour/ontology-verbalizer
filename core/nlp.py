from openai import OpenAI


class LanguageModel:
    def pseudo_to_text(self, pseudo_text: str, extra: str = None) -> str:
        return pseudo_text

    @property
    def cost(self) -> float:
        return 0.0


class ChatGptModel(LanguageModel):
    models = {
        "gpt-4o": {
            "input": 0.005,
            "output": 0.015
        },
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

    def pseudo_to_text(self, pseudo_text: str, extra: str = None) -> str:
        extra_context = extra if extra else ""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content":
                        f"""
                        You are a data expert capable of converting pseudo English sentences into a meaningful
                        and casual paragraph. Avoid repeating information.
                        {extra_context}
                        
                        Input 1:
                        x relation 1 y
                        z relation 2 x
                        m relation 3 n

                        Output 1:
                        X has this relation 1 with Y, Z shares a relation 2 with X.
                        Moreover, m has relation 3 with n.
                        
                        Input 2:
                        X is same as something that intersection of something that something that has at least 3 N and M.
                        
                        Output 2:
                        X is the same as M which has at least three N
                        """
                },
                {"role": "user", "content": pseudo_text},
            ],
            temperature=self.temperature
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


class LlamaModel(LanguageModel):
    def __init__(self, base_url, temperature=0.5):
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url,  # "http://<Your api-server IP>:port"
            api_key="sk-no-key-required"
        )

    def pseudo_to_text(self, pseudo_text: str, extra: str = None) -> str:
        response = self.client.chat.completions.create(
            model="LLaMA_CPP",
            messages=[
                {
                    "role": "system",
                    "content": "This is a conversation between User and Agent, a friendly chatbot."
                               " Agent is helpful and good at re-writing what it is told with precision and without adding any new information."
                },
                {"role": "user", "content": pseudo_text},
            ],
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    @property
    def cost(self) -> float:
        return 0.0
