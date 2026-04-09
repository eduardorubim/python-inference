import argparse

def main():
    # 1. Initialize parser with a description for the helper
    parser = argparse.ArgumentParser(description="A simple CLI tool example.")

    # 2. Add arguments with 'help' strings for the helper menu
    parser.add_argument("name", help="The name of the user to greet")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Use the arguments
    if args.verbose:
        print(f"Verbose mode is ON.")

    print(f"Hello, {args.name}")

if __name__ == "__main__":
    main()