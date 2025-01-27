import asyncio
import time
from src.llama_index import run_llama
from src.haystack import run_haystack
from src.hugging_face import run_transformers

def run_eval(task_prompt: str):
    start_time = time.time()
    asyncio.run(run_llama.run(task_prompt))
    llama_exec_time = time.time() - start_time

    start_time = time.time()
    run_haystack.run(task_prompt)
    haystack_exec_time = time.time() - start_time

    start_time = time.time()
    run_transformers.run(task_prompt)
    transformers_exec_time = time.time() - start_time

    print(f"execution times for the different frameworks:\nllama-index: {llama_exec_time}\nhaystack: {haystack_exec_time}\ntransformers: {transformers_exec_time}")

    return llama_exec_time, haystack_exec_time, transformers_exec_time

if __name__ == "__main__":

    task1 = """Please write a function called LCS that accepts two sequences and returns the longest subsequence common to the passed in sequences.
            Subsequence: A subsequence is different from a substring. The terms of a subsequence need not be consecutive terms of the original sequence.
            Example subsequence: Subsequences of "abc" = "a", "b", "c", "ab", "ac", "bc" and "abc".
            LCS examples:
            lcs( "abcdef" , "abc" ) => returns "abc"
            lcs( "abcdef" , "acf" ) => returns "acf"
            lcs( "132535365" , "123456789" ) => returns "12356"
            Notes:
            Both arguments will be strings.
            Return value must be a string.
            Return an empty string if there exists no common subsequence.
            Both arguments will have one or more characters.
            All tests should only have a single longest common subsequence. Don't worry about cases such as LCS( "1234", "3412" ), which would have two possible longest common subsequences: "12" and "34".
            Save the function to a file, debug it and finally write unittests and test it thourougly."""
    
    task2 = """Please clone this github repository: https://github.com/ossendorfluca/pete-the-baker. Have a look at the file cake.py. Based on this existing code, implement the following additional feature and save it to a new file within the cloned repository:
            Pete is now mixing the cake mixture. He has the recipe, containing the required ingredients for one cake. He also might have added some of the ingredients already, but something is missing. Can you help him to find out, what he has to add to the mixture?
            Pete only wants to bake whole cakes. And ingredients, that were added once to the mixture, can't be removed from that. That means: if he already added the amount of flour for 2.8 cakes, he needs to add at least the amount of flour for 0.2 cakes, in order to have enough flour for 3 cakes.
            If Pete already added all ingredients for an integer amount of cakes, you don't need to add anything, just return an empty hash then.
            If Pete didn't add any ingredients at all, you need to add all ingredients for exactly one cake.
            For simplicity we ignore all units and just concentrate on the numbers. E.g. 250g of flour is simply 250 (units) of flour and 1 lb of sugar is also simply 1 (unit) of sugar.
            Ingredients, which don't have to be added to the mixture (missing amount = 0), must not be present in the result.
            Examples:
            var recipe = {flour: 200, eggs: 1, sugar: 100};
            getMissingIngredients(recipe, {flour: 50, eggs: 1}); // must return {flour: 150, sugar: 100}
            getMissingIngredients(recipe, {}); // must return {flour: 200, eggs: 1, sugar: 100}
            getMissingIngredients(recipe, {flour: 500, sugar: 200}); // must return {flour: 100, eggs: 3, sugar: 100}"""
    
    task3 = """Please solve issue number 1 in this github repository: https://github.com/ossendorfluca/llm-test. 
            To solve the issue, clone the repository. 
            Then, retrieve issue number 1 and fix the described problems in the issue description by modifying the code in the cloned local repository. """

    print("Running task 1...")
    run_eval(task1)
    print("Running task 2...")
    run_eval(task2)
    print("Running task 3...")
    run_eval(task3)

