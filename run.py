import main
import argparse


def runner():
    parser = argparse.ArgumentParser(description="Run specific functions from my_module.")
    parser.add_argument('--function', type=str)
    parser.add_argument('-p', type=float)
    parser.add_argument('-n', type=int)
    parser.add_argument('-plot', action='store_true')

    args = parser.parse_args()

    if args.function == "run_batch":
        function_to_run = getattr(main, args.function, None)
        if function_to_run and callable(function_to_run):
            function_to_run()

    elif args.function is None:
        function_to_run = getattr(main, "run_model", None)
        if function_to_run and callable(function_to_run):
            function_to_run(args.p, args.n, args.plot)


if __name__ == "__main__":
    runner()
