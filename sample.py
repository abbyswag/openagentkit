#!/usr/bin/env python
"""
Heartbeat Agent Pipeline for Open WebUI
======================================
This pipeline turns an LLM into an autonomous agent that can reason in
multiple steps.  After each LLM reply the pipeline checks for a *heartbeat*
flag returned by the model.  If the model sets `{"heartbeat": true}` the
conversation continues for another iteration (up to `MAX_ITERS`).  The
agent therefore controls its own loop – while the pipeline enforces an
upper‑bound so it can never run forever.

Key design points
-----------------
* **Subclass of FunctionCallingBlueprint** – we inherit the full function
  calling helper as well as the `Tools` mechanism already used in
  Open WebUI.  All side‑effects therefore happen through tools; the LLM
  never touches the outside world directly.
* **Heartbeat contract** – the assistant must finish every turn with a
  *single* JSON object.  If that object contains `"heartbeat": true` the
  loop continues; otherwise the loop stops and the accumulated transcript
  is returned to the user.
* **Max‑iteration safety** – `MAX_ITERS` (default 5) is defined in
  `Valves` and can be overridden from the UI settings panel.  Once the
  limit is reached the loop halts even if the model still signals
  heartbeat.
* **Fully stateless between calls** – everything needed for the loop is
  rebuilt from the incoming `messages` list, so the pipeline can be
  re‑loaded or scaled horizontally without sticky sessions.

Usage
-----
1. Drop this file into your `pipelines/` folder (or a sub‑package).
2. Restart Open WebUI – it will auto‑discover the new pipeline.
3. Select **Heartbeat Agent Pipeline** from the model drop‑down.
4. Prompt the model.  If the model wants more internal reasoning it just
   replies with `{ "heartbeat": true, "thought": "…", "action": "…" }`.
   When it is ready to answer the user it omits the heartbeat flag.

Tip:  Adapt your system prompt so the model knows about this protocol,
for example:

    You are an autonomous problem‑solver.  Output **only** a JSON object
    each turn.  Include `"heartbeat": true` if you need to think or call a
    tool again; omit it when you have the final answer for the user.

"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Union, Generator, Dict, Any

import requests
from pydantic import BaseModel
from schemas import OpenAIChatMessage  # Provided by Open WebUI

from blueprints.function_calling_blueprint import (
    Pipeline as FunctionCallingBlueprint,
)
from utils.pipelines.main import get_tools_specs


class Pipeline(FunctionCallingBlueprint):
    """Heartbeat‑controlled autonomous agent pipeline."""

    # ------------------------------------------------------------------
    # Valve configuration ------------------------------------------------
    # ------------------------------------------------------------------

    class Valves(FunctionCallingBlueprint.Valves):
        """Add extra runtime‑tunable settings."""

        # Hard safety limit for the agent loop
        MAX_ITERS: int = 5

        # Name of the boolean flag inside the assistant JSON that decides
        # whether to continue.  Change this here and in your prompts if you
        # don't like the default.
        HEARTBEAT_FIELD: str = "heartbeat"

    # ------------------------------------------------------------------
    # Tools --------------------------------------------------------------
    # ------------------------------------------------------------------

    class Tools:
        """Container object for all callable tools.

        Every method with type hints and a Sphinx‑style docstring becomes a
        tool automatically, thanks to FunctionCallingBlueprint.
        """

        def __init__(self, pipeline: "Pipeline") -> None:
            self.pipeline = pipeline

        # ---- Example utility tools -----------------------------------

        def get_server_time(self) -> str:
            """Get the current UTC time on the server in ISO‑8601 format."""
            from datetime import datetime, timezone

            return datetime.now(timezone.utc).isoformat()

        def fetch_url(self, url: str) -> str:
            """Retrieve plain‑text content from *url* (GET).

            Parameters
            ----------
            url : str
                Fully qualified URL to download.
            """
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.text[:8_000]  # Safeguard: trim very long bodies

    # ------------------------------------------------------------------
    # Construction -------------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

        self.name = "Heartbeat Agent Pipeline"

        # Initialize with defaults first, then update if needed
        base_valves = super().Valves()
        self.valves = self.Valves(
            **{
                **base_valves.model_dump(),
                "pipelines": ["*"],  # Attach to every model unless filtered
            }
        )

        # Register tools
        self.tools = self.Tools(self)

    # ------------------------------------------------------------------
    # Core agent loop ----------------------------------------------------
    # ------------------------------------------------------------------

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
    ) -> Union[str, Generator[str, None, None], Iterator[str]]:
        """Main entry; returns final agent answer after internal loop."""

        max_iters: int = self.valves.MAX_ITERS
        heartbeat_field: str = self.valves.HEARTBEAT_FIELD

        # Local copy we can mutate freely
        convo: List[Dict[str, Any]] = [m.copy() for m in messages]

        # Make sure our available tools are visible to the model
        tool_specs = get_tools_specs(self.tools)

        for step in range(max_iters):
            # ------------------------------------------------------------------
            # Send current conversation to the LLM -----------------------------
            # ------------------------------------------------------------------
            payload = {
                "model": self.valves.TASK_MODEL,
                "messages": convo,
            }
            if tool_specs:
                payload["tools"] = tool_specs

            response = requests.post(
                f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=90,
            )
            response.raise_for_status()
            assistant_msg = response.json()["choices"][0]["message"]

            # Record assistant turn
            convo.append(assistant_msg)

            content: str = assistant_msg.get("content", "")

            # ------------------------------------------------------------------
            # Decide whether to continue --------------------------------------
            # ------------------------------------------------------------------
            try:
                data = json.loads(content)
                if not isinstance(data, dict):  # Not an object – stop early
                    break
                if not data.get(heartbeat_field, False):
                    # Agent signals job done
                    break
            except json.JSONDecodeError:
                # Not JSON at all – treat as final answer
                break

            # Agent wants another round – make sure the loop continues with
            # *something* so the model has a new user turn.  Convention: empty
            # string user message means "continue".
            convo.append({"role": "user", "content": ""})

        # ------------------------------------------------------------------
        # Return last assistant message content as chat completion -----------
        # ------------------------------------------------------------------
        return convo[-1]["content"].strip()
