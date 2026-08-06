"""
One-time correction of AIME question text corrupted by pdfplumber's
extraction of subscript/superscript LaTeX notation (see
src/dataset/extract_aime.py).

Discovered during the model pilot (docs/limitations.md): pdfplumber's
linear text extraction drops or mis-positions subscripts and superscripts
in the source PDFs (typeset by Po-Shen Loh's LIVE project). For simple
single exponents ("a2" for a^2) this is usually still inferable from
context; for multi-index notation (sequences a_1,...,a_n, recurrences
x_{k+1}, log bases, summation indices) it produces genuinely ambiguous or
unsolvable text, independent of which model is asked to solve it.

Rather than patch only the worst offenders, every one of the 51 curated
(Include=T) AIME questions was re-transcribed by visually reading the
source PDF pages, so the whole Hard tier has consistent, correct notation.
This corrects the SOURCE file (data/raw/aime_master.xlsx), not just the
frozen dataset, so re-running src/dataset/freeze_dataset.py reproduces the
fix rather than needing it reapplied.

Run once: python -m src.dataset.fix_aime_extraction
"""

import pandas as pd

from config.config import AIME_MASTER_PATH

# Keyed by (Year, Exam, Problem Number). Transcribed from the source PDFs
# in data/aime_questions/. Plain-text math notation (^ for exponents, /
# for fractions, sqrt() for radicals) to match the style already used
# elsewhere in the dataset.
CORRECTED_QUESTIONS = {

    # 2024 AIME I
    (2024, "I", 1): "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s + 2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at s + 1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop.",
    (2024, "I", 2): "There exist real numbers x and y, both greater than 1, such that log_x(y^x) = log_y(x^(4y)) = 10. Find xy.",
    (2024, "I", 3): "Alice and Bob play the following game. A stack of n tokens lies before them. The players take turns with Alice going first. On each turn, the player removes 1 token or 4 tokens from the stack. The player who removes the last token wins. Find the number of positive integers n less than or equal to 2024 such that there is a strategy that guarantees that Bob wins, regardless of Alice's moves.",
    (2024, "I", 4): "Jen enters a lottery by selecting 4 distinct elements of S = {1,2,3,...,9,10}. Then four elements of S are drawn at random. Jen wins a prize if at least two of her numbers were drawn, and wins the grand prize if all four of her numbers were drawn. The probability that Jen wins the grand prize given that Jen wins a prize is m/n where m and n are relatively prime positive integers. Find m + n.",
    (2024, "I", 7): "Find the largest possible real part of (75 + 117i)z + (96 + 144i)/z, where z is a complex number with |z| = 4. Here i = sqrt(-1).",
    (2024, "I", 11): "Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability that the octagon can then be rotated so that all of the blue vertices end up at positions where there had been red vertices is m/n, where m and n are relatively prime positive integers. Find m + n.",
    (2024, "I", 12): "Define f(x) = ||x| - 1/2| and g(x) = ||x| - 1/4|. Find the number of intersections of the graphs of y = 4g(f(sin(2*pi*x))) and x = 4g(f(cos(3*pi*y))).",
    (2024, "I", 13): "Let p be the least prime number for which there exists an integer n such that n^4 + 1 is divisible by p^2. Find the least positive integer m such that m^4 + 1 is divisible by p^2.",

    # 2024 AIME II
    (2024, "II", 1): "Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.",
    (2024, "II", 2): "A list of positive integers has the following properties: the sum of the items in the list is 30; the unique mode of the list is 9; the median of the list is a positive integer that does not appear in the list itself. Find the sum of the squares of all the items in the list.",
    (2024, "II", 4): "Let x, y, and z be positive real numbers that satisfy the following system of equations: log_2(x/(yz)) = 1/2, log_2(y/(xz)) = 1/3, log_2(z/(xy)) = 1/4. Then the value of |log_2(x^4 * y^3 * z^2)| is m/n where m and n are relatively prime positive integers. Find m + n.",
    (2024, "II", 6): "Alice chooses a set A of positive integers. Then Bob lists all finite nonempty sets B of positive integers with the property that the maximum element of B belongs to A. Bob's list has 2024 sets. Find the sum of the elements of A.",
    (2024, "II", 7): "Let N be the greatest four-digit integer with the property that whenever one of its digits is changed to 1, the resulting number is divisible by 7. Let Q and R be the quotient and remainder, respectively, when N is divided by 1000. Find Q + R.",
    (2024, "II", 9): "There is a collection of 25 indistinguishable white chips and 25 indistinguishable black chips. Find the number of ways to place some of these chips in a 5x5 grid such that: each cell contains at most one chip; all chips in the same row and all chips in the same column have the same color; and any additional chip placed on the grid would violate one or more of the previous two conditions.",
    (2024, "II", 11): "Find the number of triples of nonnegative integers (a,b,c) satisfying a + b + c = 300 and a^2*b + a^2*c + b^2*a + b^2*c + c^2*a + c^2*b = 6,000,000.",
    (2024, "II", 13): "Let omega ≠ 1 be a 13th root of unity. Find the remainder when the product, for k = 0 to 12, of (2 - 2*omega^k + omega^(2k)) is divided by 1000.",
    (2024, "II", 14): "Let b ≥ 2 be an integer. Call a positive integer n b-eautiful if it has exactly two digits when expressed in base b and these two digits sum to sqrt(n). For example, 81 is 13-eautiful because 81 = 63 in base 13 and 6 + 3 = sqrt(81). Find the least integer b ≥ 2 for which there are more than ten b-eautiful integers.",

    # 2025 AIME I
    (2025, "I", 1): "Find the sum of all integer bases b > 9 for which 17 (in base b) is a divisor of 97 (in base b).",
    (2025, "I", 3): "The 9 members of a baseball team went to an ice-cream parlor after their game. Each player had a single scoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let N be the number of different assignments of flavors to players that meet these conditions. Find the remainder when N is divided by 1000.",
    (2025, "I", 4): "Find the number of ordered pairs (x,y), where both x and y are integers between -100 and 100, inclusive, such that 12x^2 - xy - 6y^2 = 0.",
    (2025, "I", 5): "There are 8! = 40320 eight-digit positive integers that use each of the digits 1,2,3,4,5,6,7,8 exactly once. Let N be the number of these integers that are divisible by 22. Find the difference between N and 2025.",
    (2025, "I", 7): "The twelve letters A,B,C,D,E,F,G,H,I,J,K, and L are randomly grouped into six pairs of letters. The two letters in each pair are placed next to each other in alphabetical order to form six two-letter words, and then those six words are listed alphabetically. For example, a possible result is AB,CJ,DG,EK,FL,HI. The probability that the last word listed contains G is m/n, where m and n are relatively prime positive integers. Find m + n.",
    (2025, "I", 13): "Alex divides a disk into four quadrants with two perpendicular diameters intersecting at the center of the disk. He draws 25 more line segments through the disk, drawing each segment by selecting two points at random on the perimeter of the disk in different quadrants and connecting these two points. Find the expected number of regions into which these 27 line segments divide the disk.",
    (2025, "I", 15): "Let N denote the number of ordered triples of positive integers (a,b,c) such that a,b,c ≤ 3^6 and a^3 + b^3 + c^3 is a multiple of 3^7. Find the remainder when N is divided by 1000.",

    # 2025 AIME II
    (2025, "II", 2): "Find the sum of all positive integers n such that n + 2 divides the product 3(n + 3)(n^2 + 9).",
    (2025, "II", 4): "The product, for k = 4 to 63, of [log_k(5^(k^2-1)) / log_(k+1)(5^(k^2-4))] = [log_4(5^15)/log_5(5^12)] * [log_5(5^24)/log_6(5^21)] * [log_6(5^35)/log_7(5^32)] * ... * [log_63(5^3968)/log_64(5^3965)] is equal to m/n, where m and n are relatively prime positive integers. Find m + n.",
    (2025, "II", 7): "Suppose triangle ABC has angles angle-BAC = 84 degrees, angle-ABC = 60 degrees, and angle-ACB = 36 degrees. Let D, E, and F be the midpoints of sides BC, AC, and AB, respectively. The circumcircle of triangle DEF intersects BD, AE, and AF at points G, H, and J, respectively. The points G, D, E, H, J, and F divide the circumcircle of triangle DEF into six minor arcs, as shown. Find arc-DE + 2*arc-HJ + 3*arc-FG, where the arcs are measured in degrees.",
    (2025, "II", 8): "From an unlimited supply of 1-cent coins, 10-cent coins, and 25-cent coins, Silas wants to find a collection of coins that has a total value of N cents, where N is a positive integer. He uses the so-called greedy algorithm, successively choosing the coin of greatest value that does not cause the value of his collection to exceed N. For example, to get 42 cents, Silas will choose a 25-cent coin, then a 10-cent coin, then 7 1-cent coins. However, this collection of 9 coins uses more coins than necessary to get a total of 42 cents; indeed, choosing 4 10-cent coins and 2 1-cent coins achieves the same total value with only 6 coins. In general, the greedy algorithm succeeds for a given N if no other collection of 1-cent, 10-cent, and 25-cent coins gives a total value of N cents using strictly fewer coins than the collection given by the greedy algorithm. Find the number of values of N between 1 and 1000 inclusive for which the greedy algorithm succeeds.",
    (2025, "II", 9): "There are n values of x in the interval 0 < x < 2*pi where f(x) = sin(7*pi * sin(5x)) = 0. For t of these n values of x, the graph of y = f(x) is tangent to the x-axis. Find n + t.",
    (2025, "II", 10): "Sixteen chairs are arranged in a row. Eight people each select a chair in which to sit so that no person sits next to two other people. Let N be the number of subsets of 16 chairs that could be selected. Find the remainder when N is divided by 1000.",
    (2025, "II", 11): "Let S be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in S is an endpoint of exactly one of the 12 segments.",
    (2025, "II", 13): "Let the sequence of rationals x_1, x_2, ... be defined such that x_1 = 25/11 and x_(k+1) = (1/3) * (x_k + 1/x_k - 1) for all k ≥ 1. Then x_2025 can be expressed as m/n for relatively prime positive integers m and n. Find the remainder when m + n is divided by 1000.",
    (2025, "II", 15): "There are exactly three positive real numbers k such that the function f(x) = (x-18)(x-72)(x-98)(x-k)/x, defined over the positive real numbers, achieves its minimum value at exactly two positive real numbers x. Find the sum of these three values of k.",

    # 2026 AIME I
    (2026, "I", 1): "Patrick started walking at a constant speed along a straight road from his school to the park. One hour after Patrick left, Tanya started running at a constant speed of 2 miles per hour faster than Patrick walked, following the same straight road from the school to the park. One hour after Tanya left, Jose started bicycling at a constant speed of 7 miles per hour faster than Tanya ran, following the same straight road from the school to the park. All three people arrived at the park at the same time. The distance from the school to the park is m/n miles, where m and n are relatively prime positive integers. Find m + n.",
    (2026, "I", 2): "Find the number of positive integer palindromes written in base 10, with no zero digits, and whose digits add up to 13. For example, 42124 has these properties. Recall that a palindrome is a number whose representation reads the same from left to right as from right to left.",
    (2026, "I", 4): "Find the number of integers less than or equal to 100 that are equal to a + b + ab for some choice of distinct positive integers a and b.",
    (2026, "I", 6): "The product of all positive real numbers x satisfying the equation the-20th-root-of(x^(log_2026 x)) = 26x is an integer P. Find the number of positive integer divisors of P.",
    (2026, "I", 7): "Find the number of functions pi mapping the set A = {1,2,3,4,5,6} onto A such that for every a in A, applying pi to a six times in a row (pi(pi(pi(pi(pi(pi(a))))))) equals a.",
    (2026, "I", 8): "Let N be the number of positive integer divisors of 17017^17 that leave a remainder of 5 upon division by 12. Find the remainder when N is divided by 1000.",
    (2026, "I", 9): "Joanne has a blank fair six-sided die and six stickers each displaying a different integer from 1 to 6. Joanne rolls the die and then places the sticker labeled 1 on the top face of the die. She then rolls the die again, places the sticker labeled 2 on the top face, and continues this process to place the rest of the stickers in order. If the die ever lands with a sticker already on its top face, the new sticker is placed to cover the old sticker. Let p be the conditional probability that at the end of the process exactly one face has been left blank, given that all the even-numbered stickers are visible on faces of the die. Then p can be written as m/n, where m and n are relatively prime positive integers. Find m + n.",
    (2026, "I", 11): "The integers from 1 to 64 are placed in some order into an 8x8 grid of cells with one number in each cell. Let a_(i,j) be the number placed in the cell in row i and column j, and let M be the sum of the absolute differences between adjacent cells. That is, M = the sum, over i = 1 to 8 and j = 1 to 7, of (|a_(i,j+1) - a_(i,j)| + |a_(j+1,i) - a_(j,i)|). Find the remainder when the maximum possible value of M is divided by 1000.",
    (2026, "I", 13): "For each nonnegative integer r less than 502, define S_r = the sum, over m ≥ 0, of C(10000, 502m + r), where C(10000, n) is defined to be 0 when n > 10,000. That is, S_r is the sum of all the binomial coefficients of the form C(10000, k) for which 0 ≤ k ≤ 10,000 and k - r is a multiple of 502. Find the number of integers in the list S_0, S_1, S_2, ..., S_501 that are multiples of the prime number 503.",

    # 2026 AIME II
    (2026, "II", 1): "Find the sum of the 10th terms of all arithmetic sequences of integers that have first term equal to 4 and include both 24 and 34 as terms.",
    (2026, "II", 4): "For each positive integer n let f(n) be the value of the base-ten numeral n viewed in base b, where b is the least integer greater than the greatest digit in n. For example, if n = 72, then b = 8, and 72 as a numeral in base 8 equals 7*8 + 2 = 58; therefore f(72) = 58. Find the number of positive integers n less than 1000 such that f(n) = n.",
    (2026, "II", 5): "An urn contains n marbles. Each marble is either red or blue, and there are at least 7 marbles of each color. When 7 marbles are drawn randomly from the urn without replacement, the probability that exactly 4 of them are red equals the probability that exactly 5 of them are red. Find the sum of the five least values of n for which this is possible.",
    (2026, "II", 7): "A standard fair six-sided die is rolled repeatedly. Each time the die reads 1 or 2, Alice gets a coin; each time it reads 3 or 4, Bob gets a coin; and each time it reads 5 or 6, Carol gets a coin. The probability that Alice and Bob each receive at least two coins before Carol receives any coins can be written as m/n, where m and n are relatively prime positive integers. Find 100m + n.",
    (2026, "II", 9): "Let S denote the value of the infinite sum 1/9 + 1/99 + 1/999 + 1/9999 + .... Find the remainder when the greatest integer less than or equal to 10^100 * S is divided by 1000.",
    (2026, "II", 11): "Find the greatest integer n such that the cubic polynomial x^3 - (n/6)*x^2 + (n-11)*x - 400 has roots alpha^2, beta^2, and gamma^2, where alpha, beta, and gamma are complex numbers, and there are exactly seven different possible values for alpha + beta + gamma.",
    (2026, "II", 13): "Call finite sets of integers S and T cousins if: S and T have the same number of elements; S and T are disjoint; and the elements of S can be paired with the elements of T so that the elements in each pair differ by exactly 1. For example, {1,2,5} and {0,3,4} are cousins. Suppose that the set S has exactly 4040 cousins. Find the least number of elements the set S can have.",
    (2026, "II", 14): "For integers a and b, let a○b = a - b if a is odd and b is even, and a○b = a + b otherwise. Find the number of sequences a_1, a_2, a_3, ..., a_n of positive integers such that a_1 + a_2 + a_3 + ... + a_n = 12 and a_1○a_2○a_3○...○a_n = 0, where the operations are performed from left to right; that is, a_1○a_2○a_3 means (a_1○a_2)○a_3.",
    (2026, "II", 15): "Find the number of ordered 7-tuples (a_1, a_2, a_3, ..., a_7) having the following properties: a_k is in {1,2,3} for all k; a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 is a multiple of 3; a_1*a_2*a_4 + a_2*a_3*a_5 + a_3*a_4*a_6 + a_4*a_5*a_7 + a_5*a_6*a_1 + a_6*a_7*a_2 + a_7*a_1*a_3 is a multiple of 3.",
}


def fix_aime_extraction():

    df = pd.read_excel(AIME_MASTER_PATH)

    applied = 0
    missing = []

    for (year, exam, problem), corrected_text in CORRECTED_QUESTIONS.items():

        mask = (
            (df["Year"] == year)
            & (df["Exam"] == exam)
            & (df["Problem Number"] == problem)
        )

        if not mask.any():
            missing.append((year, exam, problem))
            continue

        df.loc[mask, "Question"] = corrected_text
        applied += 1

    df.to_excel(AIME_MASTER_PATH, index=False, sheet_name="All_Questions")

    print(f"Applied {applied}/{len(CORRECTED_QUESTIONS)} corrections.")
    if missing:
        print(f"WARNING: no matching row found for: {missing}")

    return applied, missing


if __name__ == "__main__":
    fix_aime_extraction()
