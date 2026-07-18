#!/usr/bin/env python3
"""
Project 07 Sleep Research Assistant
A command-line tool for sleep neuroscience research queries.
"""

import anthropic
import textwrap

SYSTEM = """You are an expert sleep neuroscience research 
assistant helping with Project 07 — a closed-loop BCI system
for automated lucid dream induction (LUCID: Reality?).
Be technically precise, cite relevant work, and give
actionable advice."""

def main():
    client = anthropic.Anthropic()
    history = []

    print("Sleep Research Assistant — Project 07")
    print("Type 'quit' to exit, 'clear' to reset\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'clear':
            history = []
            print("Conversation cleared.\n")
            continue
        if not user_input:
            continue

        history.append({
            "role": "user", 
            "content": user_input
        })

        response = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 500,
            system     = SYSTEM,
            messages   = history
        )

        reply = response.content[0].text
        history.append({
            "role": "assistant",
            "content": reply
        })

        print(f"\nAssistant: {textwrap.fill(reply, 70)}\n")

if __name__ == "__main__":
    main()
