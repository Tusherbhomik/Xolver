from openai import OpenAI
import json
import random
import re
import subprocess
import tempfile
import os
import pickle
import sys
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
import google.generativeai as genai
from ast_checker import ast_checker
import google.generativeai as genai

nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    # print("Downloading 'punkt_tab' resource for NLTK...")
    nltk.download('punkt_tab') 

# ========== PROMPT TEMPLATES ==========
PLANNER_PROMPT = """
You are a planner to solve a {task_type} problem. Here is the problem for which you have to plan:
{problem} and from functional calling perspective : {function}

First draft required strictly greater than {m} {task_type} specialized roles labeling "Specialized Roles" to solve the problem collaboratively with
reasoning behind your draft of each role. Format the roles clearly, for example:
Specialized Roles:
1. Role Name: Reasoning of what this agent should focus on.
2. Role Name: Reasoning...
...
m. Role Name: Reasoning...
m + 1. Role Name: Reasoning...
...

Then select exactly the highly {m} {task_type} influential roles labeling "Influential Roles" from the prior drafted "Specialized Roles" by re-checking the reasoning behind your selection and
assign the prior selected "Influential Roles" among exactly the {m} agents to solve the problem. Format the roles clearly, for example:
Influential Roles:
1. Role Name: Reasoning of what this agent should focus on.
2. Role Name: Reasoning...
...
m. Role Name: Reasoning...
"""

DYNAMIC_AGENT_PROMPT = """
You are a {role}. Your task is to solve a {task_type} problem. Here is the problem that you have to
solve:
{problem} and from functional calling perspective : {function}

You were also given a couple of similar problems to the problem above along
with their reasoning and solutions to aid you in solving the problem at hand. Here are the similar
problems you were given:
{external_retrieval}
{self_retrieval}

And here was your original response:
{prev_response}

Also here is the leading responses with execution results from the response store:
{response_store}

Think carefully about where you went wrong, relating with responses in the response store. Then, try to
fix the solution producing a thought first then reply with solution with this structure:{response_structure}
Make sure to Wrap your final response in a single ```FUNCTION_CALLING``` block.
Do NOT use any extra code block (like ```json```) and  do not add any comments inside it.
The content inside should be valid JSON that can be parsed with json.loads().

"""

VERIFIER_PROMPT = """
You are an answer extractor. Your task is to extract the answer from the response to a {task_type} problem.

The answer may be inside a code block. The code block may start with ```FUNCTION_CALLING, ```json, or just ``` with no label. It always contains a list of dictionaries with function calls and parameters.

Here is the response:
{response}

Please extract **only the content inside the first code block**, preserving the structure exactly as it appears (including nested dictionaries and lists). Do not include any other text or explanation.  

Return only the extracted content.
"""

JUDGE_PROMPT = """
You are a judge. Your task is to judge the candidate solution of a {task_type} problem. Here is the
problem for which the candidate solution you have to judge:
{problem} with function calling level definition :{function}

And here is the candidate response which to judge:
{candidate_response}

Please produce a score labeling "Score" (if the response is correct, it should be 1 otherwise should be 0) with reasoning
behind your judgement of the candidate solution to the problem.
"""

response_structure = """
Your response should be a Python list of dictionaries in the following generalized format:

[
    {
        "function_name_1": {
            "arg1": value1,  # Replace value1 with the actual argument value
            "arg2": value2,  # Replace value2 with the actual argument value
            # Add more arguments as needed
        }
    },
    {
        "function_name_2": {
            "arg1": value1,
            "arg2": value2,
            # Add more arguments as needed
        }
    },
    # Add more function calls as needed
]

Notes:
- Each element in the list corresponds to a single function call.
- The function name (dictionary key) should be replaced with the proper function name.
- Each arguments dictionary should include all required parameters for the function.
- Keep the same order of arguments as defined in the function specification.
"""

genai.configure(api_key="AIzaSyByDvR03ewMm86PoUsPDjmpMYrLn_kUBmQ")
def call_openai(messages: List[Dict[str, str]], model="gemini-2.5-flash", temperature=0.2):
    gemini_messages = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_messages.append({
            "role": role,
            "parts": [msg["content"]],
        })
    
    try:
        model_instance = genai.GenerativeModel(model_name=model)
        response = model_instance.generate_content(
            gemini_messages,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
            )
        )

        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

 
def extract_roles(planner_response: str, m: int) -> List[str]:
    roles = []

    # Match "Influential Roles" with or without **
    roles_section = re.search(
        r"\*{0,2}Influential Roles\*{0,2}:?\s*(.+?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
        planner_response,
        flags=re.S | re.I
    )
    if roles_section:
        roles_text = roles_section.group(1).strip()
        # Extract role names before colon
        numbered_roles = re.findall(r"\d+\.\s*([^:]+):", roles_text)
        if numbered_roles:
            for r in numbered_roles:
                clean = (
                    r.replace("**", "")          
                     .split("\n")[0]              
                     .replace("*Reasoning", "")   
                     .strip()
                )
                roles.append(clean)

    seen = set()
    roles = [r for r in roles if not (r in seen or seen.add(r))]

    if len(roles) < m:
        roles.extend([f"Expert Agent {i+1}" for i in range(m - len(roles))])

    return roles[:m]


def parse_score(score_str: str) -> float:
    match = re.search(r"Score:\s*([01])", score_str, flags=re.I)
    if match:
        return int(match.group(1))
    return 0 


class EpisodicMemory:
    def __init__(self, memory_file=None):
        self.memory = [] 
        self.tokenized_corpus = []
        self.bm25 = None
        self.memory_file = memory_file
        if memory_file:
            self.load_memory_safe(memory_file)

    def load_memory_safe(self, filepath: str):
        import os
        if os.path.exists(filepath) and os.stat(filepath).st_size > 0:
            try:
                self.load_memory(filepath)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not load episodic memory from {filepath} due to JSON error: {e}. Starting empty.")
                self.memory = []
                self.tokenized_corpus = []
                self.bm25 = None
            except Exception as e:
                print(f"Warning: Error loading episodic memory from {filepath}: {e}. Starting empty.")
                self.memory = []
                self.tokenized_corpus = []
                self.bm25 = None
        else:
            self.memory = []
            self.tokenized_corpus = []
            self.bm25 = None

    def load_memory(self, filepath: str):
        with open(filepath, 'r') as f:
            self.memory = json.load(f)
        self.tokenized_corpus = [word_tokenize(entry['problem'].lower()) for entry in self.memory if entry.get('problem')]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def retrieve(self, query: str, k=5) -> List[Dict[str, str]]:
      if not self.bm25:
          return []
      tokenized_query = word_tokenize(query.lower())
      scores = self.bm25.get_scores(tokenized_query)
      top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
      return [
          {"problem": self.memory[i].get("problem", ""), "solution": self.memory[i].get("solution", "")}
          for i in top_n if self.memory[i].get('solution')
      ]

    def update(self, problem: str, answer: str):
        if not problem or not answer:
            return
        self.memory.append({'problem': problem, 'solution': answer})
        self.tokenized_corpus = [word_tokenize(entry['problem'].lower()) for entry in self.memory if entry.get('problem')]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def save_memory(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.memory, f, indent=2)

def model_self_recall(problem: str, task_type: str) -> str:
    recall_prompt = f"""
    You are asked to recall from your own internal knowledge a relevant but distinct {task_type} problem labeling "Problem" and its solution labeling "Response",
    different from the following problem:
    {problem}

    Please provide the recalled problem and its complete solution.
    """

    messages = [{"role": "user", "content": recall_prompt}]
    recall_response = call_openai(messages)
    return recall_response.strip()

class SharedMemory:
    def __init__(self, m: int):
        self.memory = []  # List of dicts: agent, response, score
        self.m = m

    def update(self, new_entries: List[Dict]):
        self.memory.extend(new_entries)
        self.memory = sorted(self.memory, key=lambda x: x["score"], reverse=True)[:self.m]

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
            except ValueError:
                pass  # ignore if file is closed

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass

# ========== MAIN LOOP ==========

def main():
    agents = 2
    rounds = 1
    m = agents

    log_file = open("/home/tusher/Downloads/XOLVER/Xolver/bfcl/log.txt", "a")
    sys.stdout = Tee(sys.__stdout__, log_file)

    def read_jsonl(path):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    questions = read_jsonl("/home/tusher/Downloads/XOLVER/Xolver/bfcl/BFCL_v3_parallel.json")
    # random.shuffle(questions)

    all_responses = []

    for data in questions[:1]:
        id = data['id']
        problem = data['question'][0][0]['content']
        function = data['function']
        task_type = "function calling"

        shared_memory = SharedMemory(m)
        print(f"\n=== Solving problem ===\n{problem}\n")


        planner_prompt = PLANNER_PROMPT.format(task_type=task_type, problem=problem, m=m,function=function)
        planner_context = [{"role": "user", "content": planner_prompt}]
        planner_response = call_openai(planner_context)
        print("Planner response:\n", planner_response)

        roles = extract_roles(planner_response, m)
        if len(roles) < m:
            roles = [f"expert agent {i+1}" for i in range(m)]
        print(f"Assigned roles: {roles}")

        agent_responses = [""] * m
        judge_responses = [""] * m

        # external_retrieval = episodic_memory.retrieve(problem)
        external_retrieval = None

        for iteration in range(rounds):
            new_entries = []

            # Build response_store from shared memory
            response_store_text = "\n".join(
                f"Agent: {entry['agent']}\nResponse: {entry['response']}\nScore: {entry['score']}\n"
                for entry in shared_memory.memory
            )
            if not response_store_text:
                response_store_text = "None"

            print("=== Current Shared Memory ===")
            print(response_store_text)

            for i, role in enumerate(roles):
                prev_response = agent_responses[i]

                if external_retrieval:
                    # Format external retrieval
                    external_retrieval_text = "\n\n".join(
                        f"Problem:\n{entry['problem']}\n\nResponse:\n{entry['solution']}"
                        for entry in external_retrieval
                    )
                    self_retrieval_text = "None"
                else:
                    recall_text = model_self_recall(problem, task_type)
                    external_retrieval_text = "None"
                    self_retrieval_text = recall_text

                dynamic_prompt = DYNAMIC_AGENT_PROMPT.format(
                    role=role,
                    task_type=task_type,
                    problem=problem,
                    external_retrieval=external_retrieval_text,
                    self_retrieval=self_retrieval_text,
                    prev_response=prev_response if prev_response else "None",
                    response_store=response_store_text,
                    function=function,
                    response_structure=response_structure
                )
                messages = [{"role": "user", "content": dynamic_prompt}]
                response = call_openai(messages)
                print(f"Agent ({role}) response (iteration {iteration}):\n{response}")

                
                judge_prompt = JUDGE_PROMPT.format(task_type=task_type, problem=problem,function=function, candidate_response=response)
                score_str = call_openai([{"role": "user", "content": judge_prompt}])
                score = parse_score(score_str)
                print(f"Judge scored agent {role} response: {score} (Judge comment: {score_str[:100]}...)")

                agent_responses[i] = response
                judge_responses[i] = score_str

                shared_memory.update([{
                    "agent": role,
                    "response": response,
                    "score": score
                }])
        

        if shared_memory.memory:
            best_entry = max(shared_memory.memory, key=lambda x: x["score"])
        else:
            best_entry = {"response": agent_responses[0] if agent_responses else ""}

        verifier_prompt = VERIFIER_PROMPT.format(task_type=task_type, response=best_entry["response"])
        final_answer = call_openai([{"role": "user", "content": verifier_prompt}])
        print(f"Final Answer Extracted:\n{final_answer}")
    
        # output_file = "final_answer.json"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump({"final_answer": final_answer}, f, ensure_ascii=False, indent=4)
        # print(f"Final answer saved to {output_file}")

        # episodic_memory.update(problem, response)
        # episodic_memory.save_memory("/content/episodic_memory.json")


        # evaluation with ground truth at first extract
        result_to_save = {}
        pattern = r"```(?:FUNCTION_CALLING|json)?\s*(.*?)```"
        match = re.search(pattern, final_answer, re.DOTALL)
        if match:
            func_call_str = match.group(1).strip()
            final_extracted_result = json.loads(func_call_str)
            print(final_extracted_result)


        if final_extracted_result:
            prompt_item = function
            possible_answer_item = None
            file_path = "/home/tusher/Downloads/XOLVER/Xolver/bfcl/Possible_Answers/BFCL_v3_parallel.json"
            with open(file_path, "r") as f:
                for line in f:
                    line_data = json.loads(line)
                    if line_data["id"] == id:
                        possible_answer_item = line_data["ground_truth"]
                        break
            language = "Python"
            test_category="parallel"
            model_name="google/gemini-2.5-flash"

            # Run your AST checker
            checker_result = ast_checker(
                prompt_item,
                final_extracted_result,
                possible_answer_item,
                language,
                test_category,
                model_name,
            )
            print("="*50)
            print("Checker Result:")
            print(checker_result)
            print("="*50)
            score = 1 if checker_result.get('valid', False) else 0

            result_to_save = {
                "id": id,
                "response": final_extracted_result,
                "score": score
            }
            
            # Append to JSONL file
            output_file = "/home/tusher/Downloads/XOLVER/Xolver/bfcl/test_result.jsonl"
            with open(output_file, "a") as f:
                f.write(json.dumps(result_to_save) + "\n")

        else :
            result_to_save = {
                "id": id,                   
                "response": "Did not find proper structured response",
                "score": score
            }

        #add code to show  accuracy based  of  score over the  file where  the  final ans stored
        # with open("/home/tusher/Downloads/XOLVER/Xolver/bfcl/Results/Openrouter/parallel.jsonl", "r") as f:
        #     lines = f.readlines()
        #     total = len(lines)
        #     correct = sum(1 for line in lines if json.loads(line)["score"] == 1)
        #     accuracy = correct / total if total > 0 else 0
        #     print(f"Accuracy: {accuracy:.2f}")
        #     # print the  accuracy in a file  inside the same directory
        #     with open("/home/tusher/Downloads/XOLVER/Xolver/bfcl/Results/Openrouter/parallel_accuracy.txt", "w") as f:
        #         f.write(f"Accuracy: {accuracy:.2f}\n")
    
    

        # # Update episodic memory with new experience
        # episodic_memory.update(problem, response)
        # episodic_memory.save_memory("/home/tusher/Downloads/XOLVER/Data/retrivcod/retrivcod.json")

    
        all_responses.append({
            "problem": problem,
            "planner_response":planner_response,
            "...\ndynamic_agent_responses(upon convergence)": agent_responses,
            "...\njudge_responses(upon convergence)": judge_responses,
            "shared_memory": shared_memory.memory,
            "final_answer": final_answer
        })
    

    # Save all responses to a pickle file
    with open("/home/tusher/Downloads/XOLVER/Xolver/bfcl/bfcl_responses.pkl", "wb") as f:
        pickle.dump(all_responses, f)
    log_file.close()

if __name__ == "__main__":
    main()
