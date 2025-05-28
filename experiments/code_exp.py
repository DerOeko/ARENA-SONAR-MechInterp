def greet(name): return f"Hello, {name}!"
def add_numbers(a, b): return a + b
def subtract_numbers(a, b): return a - b
def multiply_numbers(a, b): return a * b
def divide_numbers(a, b): return a / b if b != 0 else 0
def power(base, exponent): return base ** exponent
def is_even(number): return number % 2 == 0
def is_odd(number): return number % 2 != 0
def get_length(iterable): return len(iterable)
def get_first_element(list_obj): return list_obj[0] if list_obj else None
def get_last_element(list_obj): return list_obj[-1] if list_obj else None
def capitalize_string(text): return text.capitalize()
def lowercase_string(text): return text.lower()
def uppercase_string(text): return text.upper()
def reverse_string(text): return text[::-1]
def concatenate_strings(str1, str2): return str1 + str2
def create_list_from_range(start, end): return list(range(start, end))
def get_max_from_list(numbers): return max(numbers) if numbers else None
def get_min_from_list(numbers): return min(numbers) if numbers else None
def sum_list_elements(numbers): return sum(numbers)
def average_list_elements(numbers): return sum(numbers) / len(numbers) if numbers else 0
def check_if_in_list(element, list_obj): return element in list_obj
def append_to_list(list_obj, element): return list_obj + [element]
def remove_from_list(list_obj, element): return [item for item in list_obj if item != element]
def get_unique_elements(list_obj): return list(set(list_obj))
def create_dictionary(key, value): return {key: value}
def get_dictionary_value(dict_obj, key): return dict_obj.get(key)
def add_item_to_dictionary(dict_obj, key, value): new_dict = dict_obj.copy(); new_dict[key] = value; return new_dict
def get_dictionary_keys(dict_obj): return list(dict_obj.keys())
def get_dictionary_values(dict_obj): return list(dict_obj.values())
def is_empty_list(list_obj): return not list_obj
def is_empty_string(text): return not text
def is_empty_dictionary(dict_obj): return not dict_obj
def count_occurrences(list_obj, element): return list_obj.count(element)
def sort_list_ascending(list_obj): return sorted(list_obj)
def sort_list_descending(list_obj): return sorted(list_obj, reverse=True)
def simple_true(): return True
def simple_false(): return False
def get_absolute_value(number): return abs(number)
def round_number(number, decimals=0): return round(number, decimals)
def floor_division(a, b): return a // b if b != 0 else 0
def modulo_operation(a, b): return a % b if b != 0 else 0
def boolean_and(a, b): return a and b
def boolean_or(a, b): return a or b
def boolean_not(a): return not a
def check_equality(a, b): return a == b
def check_inequality(a, b): return a != b
def check_greater_than(a, b): return a > b
def check_less_than(a, b): return a < b
def check_greater_than_or_equal(a, b): return a >= b
def check_less_than_or_equal(a, b): return a <= b
def get_type(obj): return type(obj)
def convert_to_int(value): return int(value) if isinstance(value, (int, float, str)) and str(value).replace('.', '', 1).isdigit() else 0
def convert_to_float(value): return float(value) if isinstance(value, (int, float, str)) and str(value).replace('.', '', 1).isdigit() else 0.0
def convert_to_str(value): return str(value)
def join_list_with_separator(items, separator=""): return separator.join(items)
def split_string(text, separator=" "): return text.split(separator)
def strip_whitespace(text): return text.strip()
def starts_with(text, prefix): return text.startswith(prefix)
def ends_with(text, suffix): return text.endswith(suffix)
def contains_substring(text, substring): return substring in text
def replace_substring(text, old, new): return text.replace(old, new)
def list_comprehension_squares(numbers): return [n*n for n in numbers]
def list_comprehension_evens(numbers): return [n for n in numbers if n % 2 == 0]
def dict_from_two_lists(keys, values): return dict(zip(keys, values))
def get_keys_and_values(dict_obj): return list(dict_obj.items())
def copy_list(list_obj): return list_obj[:]
def copy_dictionary(dict_obj): return dict_obj.copy()
def check_all_true(booleans): return all(booleans)
def check_any_true(booleans): return any(booleans)
def get_positive_numbers(numbers): return [n for n in numbers if n > 0]
def get_negative_numbers(numbers): return [n for n in numbers if n < 0]
def increment_number(number): return number + 1
def decrement_number(number): return number - 1
def find_index_of_element(list_obj, element): return list_obj.index(element) if element in list_obj else -1
def swap_two_elements(list_obj, index1, index2): new_list = list_obj[:]; (new_list[index1], new_list[index2]) = (new_list[index2], new_list[index1]) if 0 <= index1 < len(new_list) and 0 <= index2 < len(new_list) else None; return new_list if 0 <= index1 < len(list_obj) and 0 <= index2 < len(list_obj) else list_obj
def count_vowels(text): vowels = "aeiouAEIOU"; return sum(1 for char in text if char in vowels)
def count_consonants(text): vowels = "aeiouAEIOU"; return sum(1 for char in text if char.isalpha() and char not in vowels)
def is_palindrome(text): cleaned_text = "".join(char.lower() for char in text if char.isalnum()); return cleaned_text == cleaned_text[::-1]
def get_even_indices_elements(list_obj): return [list_obj[i] for i in range(0, len(list_obj), 2)]
def get_odd_indices_elements(list_obj): return [list_obj[i] for i in range(1, len(list_obj), 2)]
def sum_of_digits(number): return sum(int(digit) for digit in str(abs(number)))
def reverse_list(list_obj): return list_obj[::-1]
def remove_duplicates_from_string(text): return "".join(sorted(set(text), key=text.index))
def get_fibonacci_sequence(n_terms): fib = [0, 1]; return fib[:n_terms] if n_terms <= 2 else [fib.append(fib[-1] + fib[-2]) or x for x in range(n_terms - 2)] and fib
def factorial(n): return 1 if n <= 0 else n * factorial(n - 1)
def celsius_to_fahrenheit(celsius): return (celsius * 9/5) + 32
def fahrenheit_to_celsius(fahrenheit): return (fahrenheit - 32) * 5/9
def truncate_string(text, max_length): return text[:max_length] + "..." if len(text) > max_length else text
def get_current_year(): import datetime; return datetime.datetime.now().year
def get_current_month(): import datetime; return datetime.datetime.now().month
def get_current_day(): import datetime; return datetime.datetime.now().day
def check_if_leap_year(year): return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
def concatenate_lists(list1, list2): return list1 + list2
def get_intersection_of_sets(set1, set2): return list(set1.intersection(set2))
def get_union_of_sets(set1, set2): return list(set1.union(set2))
def remove_none_from_list(list_obj): return [item for item in list_obj if item is not None]
def get_character_at_index(text, index): return text[index] if 0 <= index < len(text) else ""
def string_to_list_of_chars(text): return list(text)
def get_middle_element(list_obj): return list_obj[len(list_obj)//2] if list_obj else None
def check_if_all_digits(text): return text.isdigit()
def check_if_all_alpha(text): return text.isalpha()
def count_words_in_string(text): return len(text.split())
def remove_vowels_from_string(text): return "".join(char for char in text if char.lower() not in "aeiou")
def get_tuple_from_list(list_obj): return tuple(list_obj)
def get_list_from_tuple(tuple_obj): return list(tuple_obj)
def convert_to_set(iterable): return set(iterable)
def union_of_two_lists(list1, list2): return list(set(list1) | set(list2))
def intersection_of_two_lists(list1, list2): return list(set(list1) & set(list2))
def difference_of_two_lists(list1, list2): return list(set(list1) - set(list2))
def symmetric_difference_of_two_lists(list1, list2): return list(set(list1) ^ set(list2))
def is_subset(set1, set2): return set1.issubset(set2)
def is_superset(set1, set2): return set1.issuperset(set2)
def reverse_words_in_string(text): return " ".join(text.split()[::-1])
def get_first_n_elements(list_obj, n): return list_obj[:n]
def get_last_n_elements(list_obj, n): return list_obj[-n:]
def remove_none_values(iterable): return [item for item in iterable if item is not None]
def filter_by_type(iterable, data_type): return [item for item in iterable if isinstance(item, data_type)]
def sum_of_even_numbers(numbers): return sum(n for n in numbers if n % 2 == 0)
def sum_of_odd_numbers(numbers): return sum(n for n in numbers if n % 2 != 0)
def product_of_list_elements(numbers): res = 1; [res := res * n for n in numbers]; return res
def get_even_numbers(numbers): return [n for n in numbers if n % 2 == 0]
def get_odd_numbers(numbers): return [n for n in numbers if n % 2 != 0]
def count_true_booleans(booleans): return sum(1 for b in booleans if b is True)
def count_false_booleans(booleans): return sum(1 for b in booleans if b is False)
def format_currency(amount, currency_symbol="$"): return f"{currency_symbol}{amount:.2f}"
def convert_minutes_to_hours_and_minutes(minutes): return minutes // 60, minutes % 60
def convert_hours_to_minutes(hours): return hours * 60
def get_extension_from_filename(filename): return filename.split(".")[-1] if "." in filename else ""
def remove_extension_from_filename(filename): return ".".join(filename.split(".")[:-1]) if "." in filename else filename
def is_positive(number): return number > 0
def is_negative(number): return number < 0
def is_zero(number): return number == 0
def get_random_integer(min_val, max_val): import random; return random.randint(min_val, max_val)
def get_random_float(min_val, max_val): import random; return random.uniform(min_val, max_val)
def choose_random_element(list_obj): import random; return random.choice(list_obj) if list_obj else None
def shuffle_list(list_obj): import random; new_list = list_obj[:]; random.shuffle(new_list); return new_list
def generate_random_string(length): import random, string; return ''.join(random.choice(string.ascii_letters) for _ in range(length))
def calculate_bmi(weight_kg, height_m): return weight_kg / (height_m ** 2) if height_m > 0 else 0
def convert_kg_to_pounds(kg): return kg * 2.20462
def convert_pounds_to_kg(pounds): return pounds / 2.20462
def convert_cm_to_inches(cm): return cm / 2.54
def convert_inches_to_cm(inches): return inches * 2.54
def calculate_discount(price, percentage): return price * (1 - percentage / 100)
def get_percentage(value, total): return (value / total) * 100 if total != 0 else 0
def is_leap_year(year): return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
def get_day_of_week(year, month, day): import datetime; return datetime.date(year, month, day).strftime("%A")
def days_between_dates(year1, month1, day1, year2, month2, day2): import datetime; date1 = datetime.date(year1, month1, day1); date2 = datetime.date(year2, month2, day2); return abs((date2 - date1).days)
def remove_html_tags(text): import re; return re.sub(r'<[^>]+>', '', text)
def is_valid_email(email): import re; return re.match(r"[^@]+@[^@]+.[^@]+", email) is not None
def slugify_string(text): import re; text = text.lower(); text = re.sub(r'[^a-z0-9\s-]', '', text); text = re.sub(r'\s+', '-', text); return text.strip('-')
def find_all_occurrences(text, sub): return [i for i in range(len(text)) if text.startswith(sub, i)]
def swap_case_string(text): return text.swapcase()
def is_title_case(text): return text.istitle()
def is_alpha_numeric(text): return text.isalnum()
def is_decimal(text): return text.isdecimal()
def is_space(text): return text.isspace()
def is_printable(text): return text.isprintable()
def contains_only_letters(text): return all(c.isalpha() for c in text)
def contains_only_digits(text): return all(c.isdigit() for c in text)
def format_phone_number(number_str): import re; return re.sub(r'(\d{3})(\d{3})(\d{4})', r'(\1) \2-\3', number_str)
def reverse_integer(n): return int(str(abs(n))[::-1]) * (-1 if n < 0 else 1)
def sum_of_squares(numbers): return sum(nn for n in numbers)
def product_of_squares(numbers): res = 1; [res := res * (n * n) for n in numbers]; return res
def get_common_prefix(strings): return "".join(char[0] for char in zip(*strings) if all(c == char[0] for c in char))
def group_by_first_letter(words): from collections import defaultdict; d = defaultdict(list); [d[word[0].lower()].append(word) for word in words if word]; return dict(d)
def chunk_list(list_obj, size): return [list_obj[i:i + size] for i in range(0, len(list_obj), size)]
def flatten_list_of_lists(list_of_lists): return [item for sublist in list_of_lists for item in sublist]
def get_diagonals_of_matrix(matrix): return ([matrix[i][i] for i in range(len(matrix))], [matrix[i][len(matrix)-1-i] for i in range(len(matrix))]) if matrix else ([], [])
def transpose_matrix(matrix): return [list(row) for row in zip(matrix)] if matrix else []
def calculate_dot_product(vec1, vec2): return sum(x * y for x,y in zip(vec1, vec2)) if len(vec1)==len(vec2) else 0
def get_nth_fibonacci(n): a, b = 0, 1; [a := b, b := a + b for _ in range(n)]; return a
def count_prime_numbers_up_to_n(n): return sum(1 for i in range(2, n + 1) if all(i % j != 0 for j in range(2, int(i * 0.5) + 1)))
def find_most_frequent_element(list_obj): return max(set(list_obj), key=list_obj.count) if list_obj else None
def find_least_frequent_element(list_obj): return min(set(list_obj), key=list_obj.count) if list_obj else None
def get_second_largest(numbers): return sorted(set(numbers), reverse=True)[1] if len(set(numbers)) > 1 else None
def get_second_smallest(numbers): return sorted(set(numbers))[1] if len(set(numbers)) > 1 else None
def is_perfect_square(n): return n >= 0 and (int(n0.5))**2 == n
def is_armstrong_number(n): return sum(int(digit)*len(str(n)) for digit in str(n)) == n
def get_missing_number(nums): n = len(nums); return n(n+1)//2 - sum(nums)
def is_anagram(str1, str2): return sorted(str1.lower()) == sorted(str2.lower())
def remove_duplicate_words(text): return " ".join(dict.fromkeys(text.split()))
def get_initials(name): return "".join(word[0].upper() for word in name.split())
def check_for_parentheses_balance(text): balance = 0; return all((balance := balance + (1 if char == '(' else -1 if char == ')' else 0)) >= 0 for char in text) and balance == 0
def calculate_tax(amount, rate): return amount * (rate / 100)
def apply_tax(amount, rate): return amount * (1 + rate / 100)
def get_list_of_tuples_from_dict(dict_obj): return list(dict_obj.items())
def get_dict_from_list_of_tuples(list_of_tuples): return dict(list_of_tuples)
def safe_index(list_obj, element): return list_obj.index(element) if element in list_obj else -1