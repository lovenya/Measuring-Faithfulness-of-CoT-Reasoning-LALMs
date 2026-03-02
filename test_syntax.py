import sys

_DEFAULT_SYSTEM_PROMPT = "hello"

def test():
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _DEFAULT_SYSTEM_PROMPT +
                        "CRITICAL: Respond ONLY with the conclusion tag. No conversational text.\n" + 
                        "Do not engage in conversational filler. Use the following structure:\n" +
                        " <Conclusion> \nThe answer is:[Single Letter Only in parentheses]\n</Conclusion>"
                        }],
        }
    ]

try:
    compile(open('test_syntax.py').read(), 'test_syntax.py', 'exec')
    print("Compiled successfully!")
except Exception as e:
    print(e)
