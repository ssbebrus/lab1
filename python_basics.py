from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    return sum(letter in 'aeiou' for letter in data.value.lower())


def has_unique_characters(data: TextInput) -> bool:
    return len(data.value) == len(set(data.value))


def count_one_bits(data: PositiveIntegerInput) -> int:
    return bin(data.value).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    return sum((x - y) ** 2 for x, y in zip(data.predicted, data.expected)) / len(data.predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power == 1:
            factors.append(f'({divisor})')
        elif power > 1:
            factors.append(f'({divisor}**{power})')
        divisor += 1
    if number > 1:
        factors.append(f'({number})')
    return ''.join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    total = 0
    height = 0
    while total < data.value:
        height += 1
        total += height ** 2
    return height if total == data.value else 'It is impossible'


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)
    side = (len(digits) - 1) // 2
    left = sum(int(digit) for digit in digits[:side])
    right = sum(int(digit) for digit in digits[len(digits) - side:])
    return left == right
