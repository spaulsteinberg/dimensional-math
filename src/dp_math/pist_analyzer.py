

class PistAnalyzer:

    def __init__(self, usl: float, lsl: float):
        self.usl = usl
        self.lsl = lsl

    def pist(self, values: list[float | None]) -> float:
        """
        Calculate pist. Null values are skipped.
        :param values: Values
        :return: Pist
        """
        if not values:
            raise ValueError("Must provide at lease one value.")
        passing_count = 0
        total_valid = 0
        for v in values:
            if v is None:
                continue
            passing_count += 1 if self.passing(v) else 0
            total_valid += 1
        return passing_count / total_valid if total_valid > 0 else 0

    def passing(self, value: float):
        """
        Test feature pass/fail
        :param value: Number
        :return: Passing boolean
        """
        return self.usl >= value >= self.lsl