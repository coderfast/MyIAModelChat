#!/usr/bin/env python
"""Manual tests for envAIModels endpoints"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:11434"

def test_generate():
    print("=" * 60)
    print("TEST 1: /api/generate (non-streaming)")
    print("=" * 60)
    r = requests.post(
        f"{BASE_URL}/api/generate",
        json={"prompt": "What is 2+2?", "max_tokens": 32, "temperature": 0.3},
        timeout=30
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        j = r.json()
        text = j.get("choices", [{}])[0].get("text", "")
        print(f"Generated text: {text[:100]}...")
    else:
        print(f"Error: {r.text}")
    print()


def test_chat_completions():
    print("=" * 60)
    print("TEST 2: /v1/chat/completions (non-streaming)")
    print("=" * 60)
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello, say something short"}],
            "max_tokens": 32,
            "temperature": 0.3
        },
        timeout=30
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        j = r.json()
        text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"Response: {text[:100]}...")
    else:
        print(f"Error: {r.text}")
    print()


def test_streaming():
    print("=" * 60)
    print("TEST 3: /api/generate (streaming)")
    print("=" * 60)
    r = requests.post(
        f"{BASE_URL}/api/generate",
        json={"prompt": "Hello", "max_tokens": 40, "stream": True},
        stream=True,
        timeout=30
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print("Streaming chunks:")
        for i, line in enumerate(r.iter_lines()):
            if i < 10:
                print(f"  Chunk {i}: {line[:80]}")
            if "text.complete" in str(line):
                print("  ... stream complete")
                break
    else:
        print(f"Error: {r.text}")
    print()


def test_v1_completions():
    print("=" * 60)
    print("TEST 4: /v1/completions (text completion)")
    print("=" * 60)
    r = requests.post(
        f"{BASE_URL}/v1/completions",
        json={"prompt": "The capital of France is", "max_tokens": 16, "temperature": 0.1},
        timeout=30
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        j = r.json()
        text = j.get("choices", [{}])[0].get("text", "")
        print(f"Completion: {text[:100]}...")
    else:
        print(f"Error: {r.text}")
    print()


if __name__ == "__main__":
    print("\n🚀 Starting manual endpoint tests...\n")
    try:
        test_generate()
        test_chat_completions()
        test_streaming()
        test_v1_completions()
        print("✅ All manual tests completed!")
    except Exception as e:
        print(f"❌ Error: {e}")
