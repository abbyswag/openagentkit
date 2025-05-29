from openai import OpenAI
import os
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        MODEL_ID: str = "gpt-4.1-nano"

    def __init__(self):
        self.name = "LLM Pipeline"
        self.valves = self.Valves()
        self.client = OpenAI(api_key=self.valves.OPENAI_API_KEY)

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict):
        conversation = messages + [{"role": "user", "content": user_message}]
        try:
            response = self.client.chat.completions.create(
                model=model_id or self.valves.MODEL_ID,
                messages=conversation
            )
            message = response.choices[0].message
            return message.content
        except Exception as e:
            return f"Error: {e}" 