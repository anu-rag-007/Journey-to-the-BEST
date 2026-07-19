#!/usr/bin/env python3
"""
Project 07 Sleep Research Assistant
A command-line tool for sleep neuroscience research queries.
"""

SYSTEM = """You are an expert sleep neuroscience research 
assistant helping with Project 07 � a closed-loop BCI system
for automated lucid dream induction (LUCID: Reality?).
Be technically precise, cite relevant work, and give
actionable advice."""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


class SleepResearchChat:
    """
    Multi-turn conversation with memory using Gemini.
    Maintains full conversation history.
    """

    def __init__(self, system_prompt):
        self.system = system_prompt
        self.history = []

    def chat(self, user_message):
        # Save user message
        self.history.append({
            "role": "user",
            "content": user_message
        })

        # Build conversation history
        conversation = ""

        for msg in self.history:
            if msg["role"] == "user":
                conversation += f"User: {msg['content']}"
            else:
                conversation += f"Assistant: {msg['content']}"

        # Generate response
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=conversation,
            config=types.GenerateContentConfig(
                system_instruction=self.system,
                temperature=0.4,
                max_output_tokens=500
            )
        )

        assistant_reply = response.text

        # Save assistant reply
        self.history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        return assistant_reply

    def clear(self):
        self.history = []
        print("Conversation cleared.")

    def show_history(self):
        print(f"Conversation length: {len(self.history)} turns")

        for i, msg in enumerate(self.history, 1):
            print(f"[{i}] {msg['role'].upper()}:")
            print(msg["content"][:100] + "...")
            print()

