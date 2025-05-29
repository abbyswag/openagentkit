from openai import OpenAI
import json
import os
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str
        MODEL_ID: str 

    class Tools:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        # Example tool: add two numbers
        def add(self, a: float, b: float) -> str:
            return f"The sum of {a} and {b} is {a + b}"

    def __init__(self):
        self.name = "Tool Pipeline"
        self.valves = self.Valves()
        self.tools = self.Tools(self)
        self.tool_specs = [
            {
                "name": "add",
                "description": "Add two numbers together.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "The first number."},
                        "b": {"type": "number", "description": "The second number."}
                    },
                    "required": ["a", "b"]
                }
            }
        ]
        # Set your OpenAI API key from valves
        self.client = OpenAI(api_key=self.valves.OPENAI_API_KEY)

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict):
        """
        If the user message is a function call in JSON (e.g., {"name": "add", "parameters": {"a": 2, "b": 3}}),
        call the function and return the result. Otherwise, echo the message.
        """
        # Add the user message to the conversation
        conversation = messages + [{"role": "user", "content": user_message}]
        try:
            response = self.client.chat.completions.create(
                model=model_id or self.valves.MODEL_ID,
                messages=conversation,
                functions=self.tool_specs,
                function_call="auto"
            )
            message = response.choices[0].message
            if message.function_call:
                func_call = message.function_call
                func_name = func_call.name
                params = json.loads(func_call.arguments)
                if hasattr(self.tools, func_name):
                    result = getattr(self.tools, func_name)(**params)
                    return result
                else:
                    return f"Unknown function: {func_name}"
            else:
                return message.content
        except Exception as e:
            return f"Error: {e}" 