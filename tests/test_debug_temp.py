import json
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

def test_debug():
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    with open(FIXTURES_PATH / 'gpt2_merges.txt', 'r', encoding='utf-8') as f:
        first = f.readline().rstrip()
    parts = first.split(" ")
    
    # Write debug info
    with open('debug_pytest2.txt', 'w', encoding='utf-8') as out:
        out.write(f"decoder entries: {len(gpt2_byte_decoder)}\n")
        out.write(f"U+0120 in decoder: {chr(0x0120) in gpt2_byte_decoder}\n")
        out.write(f"First merge part[0] ord: {ord(parts[0]):04X}\n")
        out.write(f"Part[0] in decoder: {parts[0] in gpt2_byte_decoder}\n")
        try:
            b = bytes([gpt2_byte_decoder[t] for t in parts[0]])
            out.write(f"Parsed OK: {b}\n")
        except KeyError as e:
            out.write(f"KeyError: {e}\n")
    
    assert parts[0] in gpt2_byte_decoder, f"Part[0] {ord(parts[0]):04X} not in decoder"
