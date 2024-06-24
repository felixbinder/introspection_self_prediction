

# read csv
import pandas as pd
# read w/o index
df = pd.read_csv("inspect_response_property_results.csv", index_col=0)
# columns predicted_correctly	task	string	response_property	meta_model	object_model
# only task == "number_triplets"

df = df[df["task"] == "number_triplets"]
# only response_property == "is_even"
df = df[df["response_property"] == "is_even"]

# print rows
# gpt-3.5-turbo-1106 corect
vanilla_df = df[df["object_model"] == "gpt-3.5-turbo-1106"]
correct_strings_vanilla: set[str] = vanilla_df[vanilla_df["predicted_correctly"] == True]["string"].to_list()
print(len(correct_strings_vanilla))
# ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2 correct\
finetuned_df = df[df["object_model"] == "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"]
correct_strings_finetuned = finetuned_df[finetuned_df["predicted_correctly"] == True]["string"].to_list()
print(len(correct_strings_finetuned))

for string in correct_strings_vanilla:
    if string not in correct_strings_finetuned:
        print(f"{string} incorrect for ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2")
    # else:
    #     print(f"ok")