atomic_fact_prompt = """Task: You will be given an English sentence. Your goal is to identify a list of atomic facts from the sentence. Atomic fact is a short sentence conveying one piece of information. Output the list of atomic facts in Python list format without giving any additional explanation.

*** Example Starts ***
Sentence: It has not yet been determined whether the negligence or willful misconduct of the Supplier caused the data breach reported on Monday.
Atomic facts: ["It has not yet been determined whether the negligence of the Supplier caused the data breach reported on Monday.", "It has not yet been determined whether the willful misconduct of the Supplier caused the data breach reported on Monday."]

Sentence: The duration and territorial scope of the license granted herein shall be defined exclusively in Annex A of this Agreement.
Atomic facts: ["The duration of the license granted herein shall be defined exclusively in Annex A of this Agreement.", "The territorial scope of the license granted herein shall be defined exclusively in Annex A of this Agreement."]
*** Example Ends ***

Sentence: {{sentence}}
Atomic facts: """
