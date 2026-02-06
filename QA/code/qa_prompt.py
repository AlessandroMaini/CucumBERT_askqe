qa_prompt = """Task: You will be given an English sentence and a list of relevant questions. Your goal is to generate a list of answers to the questions based on the sentence. Output only the list of answers in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Sentence: Tenant shall pay Base Rent on the first day of each calendar month.
Questions: ["What must the Tenant pay?", "When is the Base Rent due?"]
Answers: ["Base Rent", "On the first day of each calendar month"]

Sentence: Licensor hereby grants to Licensee a non-exclusive, non-transferable, worldwide license to use the Licensed Software solely for Licensee’s internal business purposes during the Term, subject to the conditions herein.
Questions: ["What type of license is granted by the Licensor?", "What is the permitted use of the Licensed Software?", "What is the geographic scope of the license?", "During what period is the license valid?", "What is the grant subject to?"]
Answers: ["A non-exclusive, non-transferable, worldwide license", "Solely for Licensee’s internal business purposes", "Worldwide", "During the Term", "The conditions herein"]
*** Example Ends ***

Sentence: {{sentence}}
Questions: {{questions}}
Answers: """