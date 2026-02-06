vanilla = """Task: You will be given an English sentence. Your goal is to generate a list of relevant questions based on the sentence. Output only the list of questions in Python list format without giving any additional explanation.

*** Example Starts ***
Sentence: It has not yet been determined whether the negligence or willful misconduct of the Supplier caused the data breach reported on Monday.
Questions: ["What has not yet been determined?", "Whose negligence or willful misconduct is being considered?", "What event was reported on Monday?", "What potential causes for the data breach are mentioned?"]

Sentence: The duration and territorial scope of the license granted herein shall be defined exclusively in Annex A of this Agreement.
Questions: ["Where are the duration and territorial scope of the license defined?", "What is defined exclusively in Annex A?", "What is granted herein?", "Is the definition of the scope found anywhere besides Annex A?"]
*** Example Ends ***

Sentence: {{sentence}}
Questions: """


nli = """Task: You will be given an English sentence and a list of atomic facts, which are short sentences conveying one piece of information. Your goal is to generate a list of relevant questions based on the sentence. Output the list of questions in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Sentence: It has not yet been determined whether the negligence or willful misconduct of the Supplier caused the data breach reported on Monday.
Atomic facts: ['It has not yet been determined whether the negligence of the Supplier caused the data breach reported on Monday.', 'It has not yet been determined whether the willful misconduct of the Supplier caused the data breach reported on Monday.']
Questions: ["Has it been determined whether the Supplier's negligence caused the reported data breach?", "Has it been determined whether the Supplier's willful misconduct caused the reported data breach?"]

Sentence: The duration and territorial scope of the license granted herein shall be defined exclusively in Annex A of this Agreement.
Atomic facts: ['The duration of the license granted herein shall be defined exclusively in Annex A of this Agreement.', 'The territorial scope of the license granted herein shall be defined exclusively in Annex A of this Agreement.']
Questions: ["Where shall the duration of the granted license be defined?", "Where shall the territorial scope of the granted license be defined?"]
*** Example Ends ***

Sentence: {{sentence}}
Atomic facts: {{atomic_facts}}
Questions: """


srl = """Task: You will be given an English sentence and a dictionary of semantic roles in the sentence. Your goal is to generate a list of relevant questions based on the sentence. Output the list of questions in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Sentence: It is not yet known whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.
Semantic roles: {'Verb1': {'Verb': 'known', 'ARG1': 'It', 'TMP': 'not yet', 'ARG2': 'whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19'}, 'Verb2': {'Verb': 'affects', 'ARG0': 'the severity or level of control of underlying health conditions', 'ARG1': 'the risk for severe disease associated with COVID-19'}}
Questions: ["What is not yet known?","When is it not yet known?","What is being questioned in terms of its effect on the risk for severe disease associated with COVID-19?","What affects the risk for severe disease associated with COVID-19?","What is the risk associated with COVID-19?"]

Sentence: The number of accessory proteins and their function is unique depending on the specific coronavirus.
Semantic roles: {'Verb': 'is', 'ARG1': 'The number of accessory proteins and their function', 'MNR': 'unique', 'TMP': 'depending on the specific coronavirus'}
Questions: ["What is unique depending on the specific coronavirus?", "How is the number of accessory proteins and their function described?", "When is the uniqueness of the number of accessory proteins and their function determined?"]
*** Example Ends ***

Sentence: {{sentence}}
Semantic roles: {{semantic_roles}}
Questions: """


prompts = {
    "vanilla": vanilla,
    "atomic": nli,
    "semantic": srl
}