
def is_palindrome(s):
    normalized_s = "".join(filter(str.isalnum, s)).lower()
    return normalized_s == normalized_s[::-1]
