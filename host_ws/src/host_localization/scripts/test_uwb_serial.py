#!/usr/bin/env python3
"""Quick GrowSpace listener serial sanity check.

Usage:
  python3 test_uwb_serial.py [PORT] [DURATION_SEC]
"""
import sys
import time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUWB'
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
BAUD = 115200


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(0.5)
    ser.write(b'\r\r')
    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.write(b'lep\r')
    time.sleep(0.3)

    print(f'[INFO] {PORT} @ {BAUD}, listening for {DURATION:.1f}s')
    t_end = time.time() + DURATION
    n_pos = 0
    n_other = 0
    last_pos = None
    try:
        while time.time() < t_end:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            if 'POS,' in line:
                n_pos += 1
                last_pos = line
            else:
                n_other += 1
                if n_other <= 5:
                    print(f'[OTHER] {line}')
    finally:
        ser.write(b'les\r')
        time.sleep(0.2)
        ser.close()

    print(f'[RESULT] POS lines: {n_pos}  other: {n_other}')
    if last_pos:
        print(f'[LAST]   {last_pos}')


if __name__ == '__main__':
    main()
