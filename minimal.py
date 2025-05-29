class Pipeline:
    def __init__(self):
        self.name = "Echo Pipeline"

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict):
        return f"You said: {user_message}"
