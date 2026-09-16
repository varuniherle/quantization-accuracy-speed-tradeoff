"""
dataset.py
Shared labeled dataset for zero-shot classification benchmarking.
Used by both baseline_benchmark.py and quantized_benchmark.py so both
runs are evaluated on the exact same examples.
"""

CANDIDATE_LABELS = ["world news", "sports", "business", "science and technology"]

DATA = [
    ("The United Nations held an emergency session to discuss the ongoing border dispute between the two nations.", "world news"),
    ("Protesters gathered outside the parliament building demanding political reform.", "world news"),
    ("The prime minister announced a new diplomatic agreement with neighboring countries.", "world news"),
    ("A ceasefire was declared after months of conflict in the region.", "world news"),
    ("Election officials confirmed the results after a tense recount process.", "world news"),
    ("The refugee crisis has put pressure on several governments to act.", "world news"),
    ("Leaders from around the globe met at the summit to discuss climate policy.", "world news"),
    ("A new trade deal was signed between the two countries after years of negotiation.", "world news"),
    ("The embassy issued a statement following the diplomatic incident.", "world news"),
    ("Tensions escalated after the military conducted exercises near the border.", "world news"),

    ("The striker scored a hat-trick to lead his team to victory in the final.", "sports"),
    ("The tennis champion won her fifth consecutive title this season.", "sports"),
    ("The team's coach announced his resignation after a disappointing season.", "sports"),
    ("Fans packed the stadium to watch the championship game.", "sports"),
    ("The athlete broke the world record in the 100 meter sprint.", "sports"),
    ("The basketball team clinched a playoff spot with a last second shot.", "sports"),
    ("The marathon attracted thousands of runners from around the world.", "sports"),
    ("The referee's controversial decision sparked outrage among fans.", "sports"),
    ("The cricket match was postponed due to heavy rain.", "sports"),
    ("The boxer successfully defended his title in a closely contested match.", "sports"),

    ("The company reported a significant increase in quarterly profits.", "business"),
    ("Stock prices tumbled after the earnings report missed expectations.", "business"),
    ("The startup secured a new round of funding from venture capital firms.", "business"),
    ("The central bank raised interest rates to curb inflation.", "business"),
    ("The merger between the two firms was approved by regulators.", "business"),
    ("Shares of the tech giant hit an all time high this week.", "business"),
    ("The retailer announced plans to close several underperforming stores.", "business"),
    ("Unemployment figures dropped to their lowest level in a decade.", "business"),
    ("The CEO stepped down amid an accounting scandal investigation.", "business"),
    ("Consumer spending rose sharply during the holiday shopping season.", "business"),

    ("Researchers developed a new algorithm that improves image recognition accuracy.", "science and technology"),
    ("The space agency successfully launched a satellite into orbit.", "science and technology"),
    ("Scientists discovered a new species of deep sea fish.", "science and technology"),
    ("The tech company unveiled its latest smartphone with an improved camera.", "science and technology"),
    ("A breakthrough in battery technology could extend electric vehicle range.", "science and technology"),
    ("Engineers designed a more efficient solar panel using new materials.", "science and technology"),
    ("The study published in a leading journal sheds light on brain function.", "science and technology"),
    ("A new vaccine candidate showed promising results in early trials.", "science and technology"),
    ("The rover sent back new images of the planet's surface.", "science and technology"),
    ("Developers released an update that fixes several security vulnerabilities.", "science and technology"),
]

TEXTS = [t for t, _ in DATA]
TRUE_LABELS = [l for _, l in DATA]