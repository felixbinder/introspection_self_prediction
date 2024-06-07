import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals.utils import run_command


def fn(x):
    return x


def main():
    lrs = [0.1, 0.33, 3, 10]
    epochs = [2, 4]

    command_list = []
    # use multiprocessing to limit to 3 concurrent calls
    for lr in lrs:
        for epoch in epochs:
            command = f"python -m evals.run_finetuning study_name=training_on_everything_may_15_w_gemini/gemini-1.0-pro-002 language_model=gemini-1.0-pro-002 learning_rate={lr} notes=LR{lr}E{epoch} epochs={epoch}"
            command_list.append(command)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for command in command_list:
            futures[executor.submit(run_command, command)] = command
            # futures[executor.submit(fn, command)] = command

        for future in as_completed(futures):
            try:
                print(f"Command {futures[future]} with result {future.result()} ")
            except Exception:
                print(f"Command {futures[future]} with exception: \n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
