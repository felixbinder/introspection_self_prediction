import numpy as np

from evals.analysis.string_cleaning import clean_string

VOCAB_SIZE = 50257  # gpt-3.5-turbo—using the same vocab size for all models


def exclude_noncompliant(df):
    df = df.copy()
    if "compliance_meta" in df.columns and "compliance_object" in df.columns:
        df = df[(df["compliance_meta"] == True) & (df["compliance_object"] == True)]  # noqa: E712
    else:
        df = df[df["compliance"] == True]  # noqa: E712
    return df


def calc_accuracy(df):
    """Calculate the accuracy of the model"""
    df = exclude_noncompliant(df)
    return (
        df["extracted_property_meta"].apply(clean_string) == df["extracted_property_object"].apply(clean_string)
    ).mean()


def calc_accuracy_with_excluded(df):
    """What is the accuracy if we count non-compliance as wrong answers?"""
    df["correct"] = df["extracted_property_meta"].apply(clean_string) == df["extracted_property_object"].apply(
        clean_string
    )
    df["correct"] = df["correct"] & (df["compliance_meta"] == True)  # noqa: E712
    return df["correct"].mean()


def get_samples_for_calc_accuracy_with_excluded(df):
    """What is the accuracy if we count non-compliance as wrong answers? This function returns the individual pairs of samples."""
    df["correct"] = df["extracted_property_meta"].apply(clean_string) == df["extracted_property_object"].apply(
        clean_string
    )
    df["correct"] = df["correct"] & (df["compliance_meta"] == True)  # noqa: E712
    return df["correct"]


def get_samples_for_calc_accuracy(df):
    """Calculate the accuracy of the model. This function returns the individual pairs of samples."""
    df = exclude_noncompliant(df)
    return df["extracted_property_meta"].apply(clean_string) == df["extracted_property_object"].apply(clean_string)


def bootstrap_accuracy_ci(df, num_bootstraps=1000, ci=95):
    df = df[(df["compliance_meta"] == True) & (df["compliance_object"] == True)]  # noqa: E712

    bootstrap_accuracies = []

    # Resampling the data frame with replacement and calculating accuracies
    for _ in range(num_bootstraps):
        resampled_df = df.sample(n=len(df), replace=True)
        accuracy = calc_accuracy(resampled_df)
        bootstrap_accuracies.append(accuracy)

    # Calculating the lower and upper percentiles
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    ci_lower = np.percentile(bootstrap_accuracies, lower_percentile)
    ci_upper = np.percentile(bootstrap_accuracies, upper_percentile)

    return ci_lower, ci_upper


def baseline_accuracy_under_mode(df, base_df):
    """What would be the accuracy if the model always picked the most common response in the base responses?"""
    df = exclude_noncompliant(df)
    mode = base_df["response"].mode()[0]
    accuracy_under_mode = (df["extracted_property_object"] == mode).mean()
    return accuracy_under_mode


def baseline_accuracy_under_distribution(df, base_df, iters=100):
    """What would be the accuracy if the model always picked a response from the base responses according to the distribution?"""
    df = exclude_noncompliant(df)
    accuracies = []
    for _ in range(iters):
        # randomly sample from the base responses
        sample = np.random.choice(base_df["response"], len(df), replace=True)
        accuracy = (df["extracted_property_object"] == sample).mean()
        accuracies.append(accuracy)
    return np.mean(accuracies)


def likelihood_of_correct_first_token(df):
    """What is the log likelihood of the correct answer under the first token distribution of the meta response?"""

    def first_log_prob_likelihood(row):
        try:
            # get logprobs for first token
            logprobs = row["logprobs_meta"]
            if isinstance(logprobs, str):
                logprobs = eval(logprobs)
            try:
                if logprobs is None or np.isnan(logprobs):
                    return None
            except TypeError:
                pass  # it wasn't np.nan then
            logprobs = logprobs[0]  # only the first token
            if logprobs is None or len(logprobs) == 0:
                return None
            target = row["extracted_property_object"]
            target = str(target)  # in case it's a number
            for token, log_prob in logprobs.items():
                if clean_string(target).startswith(clean_string(token)):
                    return log_prob
            # if the log prob is not in the top n, so we calculate the flat probability of the outside of the top n
            top_n_mass = np.sum([v for k, v in logprobs.items()])
            outside_top_n_mass = 1 - top_n_mass
            return np.log(outside_top_n_mass / (VOCAB_SIZE - len(logprobs)))
        except IndexError:
            return None

    df = exclude_noncompliant(df)
    return df.apply(first_log_prob_likelihood, axis=1).dropna().mean()
