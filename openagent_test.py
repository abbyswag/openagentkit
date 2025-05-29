from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint
from openai_function_calling import tool  # needed for decorators

class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        pass

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

        @tool(name="calculate", description="Performs arithmetic operation between two numbers")
        def calculate(self, operation: str, a: float, b: float) -> str:
            if operation == 'add':
                return f"The sum of {a} and {b} is {a + b}"
            elif operation == 'subtract':
                return f"The difference between {a} and {b} is {a - b}"
            elif operation == 'multiply':
                return f"The product of {a} and {b} is {a * b}"
            elif operation == 'divide':
                if b == 0:
                    return "Cannot divide by zero"
                return f"The quotient of {a} and {b} is {a / b}"
            else:
                return f"Unknown operation: {operation}"

    def __init__(self):
        super().__init__()

        self.name = "openagent_test"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],
            },
        )
        self.tools = self.Tools(self)

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict):
        """
        Handles chat messages and invokes function-calling tools if needed.
        """
        # Use the function-calling logic from the blueprint
        # Get the last user message (or use the provided one)
        last_user_message = user_message or (messages[-1]["content"] if messages else "")

        # Get the tools specs (for OpenAI function calling)
        from utils.pipelines.main import get_tools_specs
        import json
        tools_specs = get_tools_specs(self.tools)

        # Prepare the system prompt
        prompt = self.prompt.format(json.dumps(tools_specs, indent=2))
        content = "History:\n" + "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages[::-1][:4]]
        ) + f"\nQuery: {last_user_message}"

        # Run the OpenAI completion to see if a function should be called
        result = self.run_completion(prompt, content)
        messages_with_function = self.call_function(result, messages)

        # Return the latest assistant/system message as the response
        for msg in reversed(messages_with_function):
            if msg["role"] in ("assistant", "system"):
                return msg["content"]
        # Fallback: echo the user message
        return f"You said: {last_user_message}"
