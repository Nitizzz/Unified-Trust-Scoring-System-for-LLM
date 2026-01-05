import re

def clean_line(line):
    # Remove ANSI escape sequences
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    # Handle \r: keep only the text after the last \r if present
    if '\r' in line:
        line = line.split('\r')[-1]
    return line.strip()

try:
    with open('training.log', 'r', encoding='utf-16') as f:
        lines = f.readlines()
        for line in lines[-50:]:
            print(clean_line(line))
except Exception as e:
    print(f"Error: {e}")
