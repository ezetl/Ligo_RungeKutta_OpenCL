# coding: utf-8
import sys
from ode45 import Ode45

def main():
    ode45 = Ode45(batch=3)
    ode45.execute()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulación interrumpida.\n")
    sys.exit(0)
