atomic_fact_prompt = """Task: You will be given an English sentence. Your goal is to identify a list of atomic facts from the sentence. Atomic fact is a short sentence conveying one piece of information. Output the list of atomic facts in Python list format without giving any additional explanation.

*** Example Starts ***
Sentence: It is not yet known whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.
Atomic facts: ["It is not yet known whether the severity of underlying health conditions affects the risk for severe disease associated with COVID-19.", "It is not yet known whether the level of control of underlying health conditions affects the risk for severe disease associated with COVID-19."]

Sentence: The number of accessory proteins and their function is unique depending on the specific coronavirus.
Atomic facts: ["The number of accessory proteins is unique depending on the specific coronavirus.", "The function of accessory proteins is unique depending on the specific coronavirus."]
*** Example Ends ***

Sentence: {{sentence}}
Atomic facts: """
