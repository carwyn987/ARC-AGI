from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from helper import load_json_data, compare_grids, get_grid_list, get_grid_json

class ARC_AGI_Format(BaseModel):
    output: list[list[int]] = Field(description="2d list representing the solution to the ARC task input")

llm = OpenAI()
chat_model = ChatOpenAI(model="gpt-4o") # gpt-4o gpt-3.5-turbo-0125 gpt-3.5-turbo
parser = JsonOutputParser(pydantic_object=ARC_AGI_Format)

def generate_query(data, verbose=False):
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

    # assert len(test_inputs) == 1, "Only one test input is supported"
    test_inputs = [test_inputs[0]]
    test_answers = [test_answers[0]]

    # combine the query, the training examples, and the test input(s) into one single prompt
    query = f"{query}\n\nTraining Examples:\n"
    for i, (input_grid, output_grid) in enumerate(training_data):
        query += f"Training Example {i+1}:\n"
        query += "Input:\n"
        query += get_grid_list(input_grid)
        query += "\n"
        query += "Output:\n"
        query += get_grid_list(output_grid)
        query += "\n"

    query += "Test Input:\n"
    query += get_grid_list(test_inputs[0])
    query += "\n"

    query += "What is the solution to the ARC task test input?"

    if verbose:
        print(query)

    return query

def get_model_output(query):
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_model | parser

    output = chain.invoke({"query": query})

    return output