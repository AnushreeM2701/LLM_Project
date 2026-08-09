"""
Error-type and word meanings shown in static figures. Mirrors the
dashboard's ERROR_TYPE_DEFINITIONS / WORD_DEFINITIONS verbatim.
"""

ERROR_TYPE_DEFINITIONS = {
    "Arithmetic Error": "A plain calculation slip - wrong addition/multiplication/etc.",
    "Algebraic Manipulation Error": "Mistake while simplifying, expanding, or solving an equation (e.g. sign error).",
    "Probability Reasoning Error": "Misapplied independence, conditional probability, or normalization.",
    "Combinatorial Counting Error": "Over- or under-counting; wrong permutation vs. combination choice.",
    "Number Theory Error": "Wrong handling of divisibility, modular arithmetic, parity, or integer properties.",
    "Formula Misapplication": "Used a real formula, but the wrong one for this situation.",
    "Incorrect Assumption": "Reasoning relied on something not actually given or true in the problem.",
    "Logical Reasoning Error": "A flaw in the deduction itself, not tied to a specific math domain.",
    "Incomplete Reasoning": "Stopped short - didn't finish the steps needed to justify the answer.",
    "Answer Extraction Error": "The reasoning was fine, but the final answer couldn't be parsed out of it.",
    "Other": "Catch-all for anything that doesn't fit the categories above.",
}

WORD_DEFINITIONS = {
    "assumed": "Treated something as true without justifying it.",
    "guessed": "Picked a value or answer without deriving it.",
    "failed": "Didn't complete or find something the solution needed.",
    "missed": "Overlooked a case, value, or condition entirely.",
    "counted": "A counting mistake specifically.",
    "counts": "A counting mistake specifically.",
    "count": "A counting mistake specifically.",
    "analyzing": "The error happened during an analysis step.",
    "sum": "Error in a summation step.",
    "solutions": "Error in counting or finding solutions to an equation.",
    "number": "Error in counting or finding solutions to an equation.",
    "probability": "Error in a probability calculation.",
    "expansion": "Error expanding an algebraic or modular expression.",
    "condition": "Error in a divisibility or derived condition.",
    "intersection": "Error counting or finding intersections (curves, chords).",
    "modulo": "Error in a modular arithmetic step.",
    "arithmetic": "Error in a modular arithmetic step.",
    "maximum": "Error in a maximum-value computation.",
    "cycle": "Error in permutation cycle-counting.",
    "symmetry": "Error in a symmetry argument.",
    "equals": "Error in an equation or equality step.",
    "values": "Error involving specific numeric values.",
    "random": "Error involving random sampling/testing, or a randomness-based concept.",
    "colorings": "Error counting/enumerating valid colorings (combinatorics).",
    "congruences": "Error solving modular congruences.",
    "formula": "Used the wrong formula, or misapplied a real one.",
    "lengths": "Error involving geometric lengths (e.g. arc or chord lengths).",
    "valid": "Error in determining which cases/values are actually valid.",
}
