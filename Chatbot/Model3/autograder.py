# # Not working
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
# pipe = pipeline(
#     "text-generation", model=model, tokenizer=tokenizer
# )
# hf = HuggingFacePipeline(pipeline=pipe)
prompt_template_text = """You are a high school history teacher grading
                         homework assignments. Based on homework question indicated by "**Q:**"
                         and correct answer indicated by "**A:**", your task is to determine
                         whether the student's answer is correct in yes or no. Grading is binary; therefore,
                         student's answers can be correct or wrong. Simple misspelling are okay.
                        
                         **Q:** {question}
                         **A:** {correct_answer}
                        
                         **Student's Answer:** {student_answer}
                         """
question = "Who was the 35th president of the United States of America?"
correct_answer = "John F. Kennedy"
student_answer_list = ["John F. Kennedy", "JFK", "FDR", "John F. Kenedy",
                       "John Kennedy", "Jack Kennedy", "Jacquelin Kennedy",
                       "Robert F. Kenedy"]

for student_answer in student_answer_list:
    messages = [
        SystemMessage(content=prompt_template_text),
        HumanMessage(content=[{'question':question}, {'correct_answer':correct_answer}, {'student_answer':student_answer}]),
    ]
    chain = model | StrOutputParser()
    chain.invoke(messages)

# for student_answer in student_answer_list:
#     formatted_text = prompt_template_text.format(
#         question=question,
#         correct_answer=correct_answer,
#         student_answer=student_answer
#     )
#     sequences = pipe(formatted_text, return_full_text = False, max_new_tokens = 15)
#     for seq in sequences:
#         print(f"Result: {seq['generated_text']}")

    
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
# from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain.schema import BaseOutputParser, OutputParserException, StrOutputParser
# from typing import Optional

# class GradeOutputParser(BaseOutputParser):
#     def parse(self, text: str) -> Optional[str]:
#         if "wrong" in text.lower():
#             raise OutputParserException("The answer is incorrect.")
#         elif "correct" in text.lower():
#             return "The answer is correct."

# pipe = HuggingFacePipeline(
#     'text-generation',
#     model=HuggingFaceEndpoint(
#         repo_id="TinyLlama/TinyLlama_v1.1",
#         huggingfacehub_api_token="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",
#         max_new_tokens=250,
#         return_full_text=False,
#         watermark=False,
#         stop_sequences=[],
#         timeout=300
#     ),
#     output_parser=GradeOutputParser()

# )

# # model = HuggingFaceEndpoint(
# #     temperature=0.5, 
# #     repo_id="TinyLlama/TinyLlama_v1.1", 
# #     huggingfacehub_api_token="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",
# #     max_new_tokens=250,
# #     return_full_text = False,
# #     watermark = False,
# #     stop_sequences = [],
# #     timeout = 300
# # )

# prompt_template_text = """You are a high school history teacher grading
#                         homework assignments. Based on homework question indicated by "**Q:**"
#                         and correct answer indicated by "**A:**", your task is to determine
#                         whether the student's answer is correct. Grading is binary; therefore,
#                         student's answers can be correct or wrong. Simple misspelling are okay.
                        
#                         **Q:** {question}
#                         **A:** {correct_answer}
                        
#                         **Student's Answer:** {student_answer}
#                         """


# sequences = pipe(
#     prompt_template_text,
#     max_new_tokens = 5
# )

# prompt = PromptTemplate(
#             input_variables=["question", "correct_answer", "student_answer"],
#             template=prompt_template_text)

# prompt1 = ChatPromptTemplate.from_messages([
#     ("system", """You are a high school history teacher grading
#                         homework assignments. Based on homework question indicated by "**Q:**"
#                         and correct answer indicated by "**A:**", your task is to determine
#                         whether the student's answer is correct. Grading is binary; therefore,
#                         student's answers can be correct or wrong. Simple misspelling are okay."""),
#     ("user", """ **Q:** {question},
#                         **A:** {correct_answer},
                        
#                         **Student's Answer:** {student_answer}
#                         """)
# ])

# question = "Who was the 35th president of the United States of America?"
# correct_answer = "John F. Kennedy"
# student_answer_list = ["John F. Kennedy", "JFK", "FDR", "John F. Kenedy",
#                        "John Kennedy", "Jack Kennedy", "Jacquelin Kennedy",
#                        "Robert F. Kenedy"]

# for student_answer in student_answer_list:
#     # input_dict = {'question': question, 'correct_answer': correct_answer, 'student_answer': student_answer}
#     # prompt_value = PromptTemplate([question, correct_answer, student_answer])
#     # prompt_value = prompt1.invoke({'question': question, 'correct_answer': correct_answer, 'student_answer': student_answer})
#     chain = model | prompt1 | StrOutputParser
#     result = chain.invoke({'question': question, 'correct_answer': correct_answer, 'student_answer': student_answer})
#     print(student_answer + " - " + str(result))
#     print('\n')
