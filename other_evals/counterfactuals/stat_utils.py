import statsmodels.stats.api
from pydantic import BaseModel
from slist import Slist


class ConfidenceIntervalDescription(BaseModel):
    average: float
    lower: float
    upper: float
    count: int

    def formatted(self) -> str:
        return f"{self.average:.3f}+-{self.upper-self.average:.2f}, n={self.count}"

    def ci_string(self) -> str:
        return f"{self.average:.3f}+-{self.upper-self.average:.2f}"


def average_with_95_ci(data: Slist[bool]) -> ConfidenceIntervalDescription:
    average = data.average_or_raise()
    # calculate the 95% confidence interval
    lower, upper = statsmodels.stats.api.proportion_confint(data.sum(), data.length, alpha=0.05, method="wilson")
    return ConfidenceIntervalDescription(average=average, lower=lower, upper=upper, count=data.length)  # type: ignore
