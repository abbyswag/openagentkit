from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint


class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        # Optional: customize valves
        pass

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

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
                "pipelines": ["*"],  # Connect to all pipelines
            },
        )
        self.tools = self.Tools(self)
