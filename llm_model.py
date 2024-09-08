from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

class ARC_AGI_Format(BaseModel):
    output: list[list[int]] = Field(description="2d list representing the solution to the ARC task input")

    # def set_size(self, size: int):
    #     # size probably needs to be the one dimensions of the 2d list
    #     self.output.min_items = size
    #     self.output.max_items = size


llm = OpenAI()
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125")
parser = JsonOutputParser(pydantic_object=ARC_AGI_Format)

# GENERATING QUERY
import json
with open("data/evaluation/ef26cbf6.json", "r") as file:
    data = json.loads(file.read())

query = "What is the solution to the ARC task input? ARC consists of access to the training examples for the \
task (both the input and output grids), as well as the input grid of the test examples for the \
task. The test-taker must construct on its own the output grid corresponding to the input grid \
of each test example. “Constructing the output grid” is done entirely from scratch, meaning \
that the test-taker must decide what the height and width of the output grid should be, what \
symbols it should place on the grid, and where. The task is successfully solved if the test- \
taker can produce the exact correct answer on all test examples for the task (binary measure \
of success)"

training_inputs = [data["train"][i]["input"] for i in range(len(data["train"]))]
training_answers = [data["train"][i]["output"] for i in range(len(data["train"]))]
training_data = zip(training_inputs, training_answers)

test_inputs = [data["test"][i]["input"] for i in range(len(data["test"]))]
test_answers = [data["test"][i]["output"] for i in range(len(data["test"]))]

assert len(test_inputs) == 1 and len(test_answers) == 1

# combine the query, the training examples, and the test input(s) into one single prompt
query = f"{query}\n\nTraining Examples:\n"
for i, (input_grid, output_grid) in enumerate(training_data):
    query += f"Training Example {i+1}:\n"
    for row in input_grid:
        query += " ".join(str(x) for x in row) + "\n"
    query += "\n"
    for row in output_grid:
        query += " ".join(str(x) for x in row) + "\n"
    query += "\n"

query += "Test Input:\n"
for row in test_inputs[0]:
    query += " ".join(str(x) for x in row) + "\n"
query += "\n"

query += "What is the solution to the ARC task test input?"


prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | chat_model | parser

chain.invoke({"query": query})