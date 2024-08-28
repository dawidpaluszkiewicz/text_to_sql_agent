zero_shot_cot_prompt = "Let's think step by step."
plan_and_solve_cot_prompt = """Let’s first understand the problem and
devise a plan to solve the problem. Then, let’s
carry out the plan and solve the problem step by
step."""
plan_and_solve_cot_prompt_v2 = """Let's approach this SQL problem using the following steps:
1. Understand the Problem:
   - Identify the question we need to answer
   - Determine the relevant tables and their relationships
   - Note any specific conditions or constraints

2. Plan the Query:
   - Outline the main components of the SQL query (SELECT, FROM, JOIN, WHERE, GROUP BY, etc.)
   - Determine the logical order of operations
   - Consider any necessary subqueries or complex joins

3. Solve Step-by-Step:
   - Write each part of the query sequentially
   - Explain the purpose and function of each clause
   - Address any potential performance considerations

4. Review and Optimize:
   - Check if the query answers the original question
   - Look for opportunities to simplify or optimize the query
   - Consider alternative approaches if applicable

5. Provide Final Query:
   - Present the complete, optimized SQL query
   - Briefly explain what the query does and how it solves the original problem
"""
