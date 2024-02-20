from openai import OpenAI


class LanguageModel:
    def pseudo_to_text(self, pseudo_text: str) -> str:
        return pseudo_text


class ChatGptModel(LanguageModel):
    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo-0613', temperature=0.5):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)

    def pseudo_to_text(self, pseudo_text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content":
                        """
                        You are a data expert capable of converting simplified relation triples into a meaningful paragraph
                        without losing any meaning or adding anything new. Group information based on the assigned groups.

                        Example:
                        <group_1>
                            <group_2>
                                x-[relation 1]->y
                                z-[relation 2]->x
                            </group_2>
                            m-[relation 3]->n
                        </group_1>

                        Output:
                        X has this relation 1 with Y, Z shares a relation 2 with X.
                        Moreover, m has relation 3 with n.
                        """
                },
                {"role": "user", "content": pseudo_text},
            ]
        )
        return response.choices[0].message.content
