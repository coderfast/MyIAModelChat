import traceback
try:
    with open('data_preparer.py') as f:
        code = f.read()
    compile(code, 'data_preparer.py', 'exec')
    print("No syntax errors found!")
except SyntaxError as e:
    print(f"❌ Syntax Error on Line {e.lineno}: {e.msg}")
    if e.text:
        print(f"Text: {e.text}")
    if e.offset:
        print(f"Offset: {' ' * (e.offset - 1)}^")
    print("\nFull traceback:")
    traceback.print_exc()
